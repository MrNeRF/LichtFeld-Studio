/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_renderer.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include <cassert>
#include <glad/glad.h>
#include <glm/gtc/matrix_inverse.hpp>

namespace lfs::rendering {

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

        // Position (location 0)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_.get());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);

        // Normal (location 1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_normals_.get());
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);

        // Tangent (location 2)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_tangents_.get());
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(2);

        // Texcoord (location 3)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoords_.get());
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
        glEnableVertexAttribArray(3);

        // Color (location 4)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_.get());
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(4);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_.get());

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        initialized_ = true;
        LOG_INFO("MeshRenderer initialized");
        return {};
    }

    Result<void> MeshRenderer::setupFBO(int width, int height) {
        assert(width > 0 && height > 0);

        // Color texture (RGBA8)
        GLuint color_tex;
        glGenTextures(1, &color_tex);
        glBindTexture(GL_TEXTURE_2D, color_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        color_texture_ = Texture(color_tex);

        // Depth texture (R32F for view-space depth, compositing)
        GLuint depth_tex;
        glGenTextures(1, &depth_tex);
        glBindTexture(GL_TEXTURE_2D, depth_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        depth_texture_ = Texture(depth_tex);

        // Depth renderbuffer for actual depth testing
        GLuint depth_rb;
        glGenRenderbuffers(1, &depth_rb);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
        depth_rbo_ = RBO(depth_rb);

        // Framebuffer
        GLuint fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture_.get(), 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, depth_texture_.get(), 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo_.get());

        GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, draw_buffers);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return std::unexpected("Mesh FBO incomplete");
        }

        fbo_ = FBO(fbo);
        fbo_width_ = width;
        fbo_height_ = height;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return {};
    }

    void MeshRenderer::resize(int width, int height) {
        if (width == fbo_width_ && height == fbo_height_)
            return;

        auto result = setupFBO(width, height);
        if (!result) {
            LOG_ERROR("Failed to resize mesh FBO: {}", result.error());
        }
    }

    Result<void> MeshRenderer::uploadMeshData(const lfs::core::MeshData& mesh) {
        if (mesh.vertex_count() == uploaded_vertex_count_ &&
            mesh.face_count() == uploaded_face_count_) {
            return {};
        }

        auto cpu_verts = mesh.vertices.to(lfs::core::Device::CPU).contiguous();
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_.get());
        glBufferData(GL_ARRAY_BUFFER,
                     cpu_verts.numel() * sizeof(float),
                     cpu_verts.ptr<float>(), GL_DYNAMIC_DRAW);

        if (mesh.has_normals()) {
            auto cpu_normals = mesh.normals.to(lfs::core::Device::CPU).contiguous();
            glBindBuffer(GL_ARRAY_BUFFER, vbo_normals_.get());
            glBufferData(GL_ARRAY_BUFFER,
                         cpu_normals.numel() * sizeof(float),
                         cpu_normals.ptr<float>(), GL_DYNAMIC_DRAW);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_normals_.get());
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        }

        if (mesh.has_tangents()) {
            auto cpu_tangents = mesh.tangents.to(lfs::core::Device::CPU).contiguous();
            glBindBuffer(GL_ARRAY_BUFFER, vbo_tangents_.get());
            glBufferData(GL_ARRAY_BUFFER,
                         cpu_tangents.numel() * sizeof(float),
                         cpu_tangents.ptr<float>(), GL_DYNAMIC_DRAW);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_tangents_.get());
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        }

        if (mesh.has_texcoords()) {
            auto cpu_tc = mesh.texcoords.to(lfs::core::Device::CPU).contiguous();
            glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoords_.get());
            glBufferData(GL_ARRAY_BUFFER,
                         cpu_tc.numel() * sizeof(float),
                         cpu_tc.ptr<float>(), GL_DYNAMIC_DRAW);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoords_.get());
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        }

        if (mesh.has_colors()) {
            auto cpu_colors = mesh.colors.to(lfs::core::Device::CPU).contiguous();
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_.get());
            glBufferData(GL_ARRAY_BUFFER,
                         cpu_colors.numel() * sizeof(float),
                         cpu_colors.ptr<float>(), GL_DYNAMIC_DRAW);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_.get());
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        }

        auto cpu_idx = mesh.indices.to(lfs::core::Device::CPU).contiguous();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_.get());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     cpu_idx.numel() * sizeof(int32_t),
                     cpu_idx.ptr<int32_t>(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        uploaded_vertex_count_ = mesh.vertex_count();
        uploaded_face_count_ = mesh.face_count();

        return {};
    }

    Result<void> MeshRenderer::render(const lfs::core::MeshData& mesh,
                                      const glm::mat4& model,
                                      const glm::mat4& view,
                                      const glm::mat4& projection,
                                      const glm::vec3& camera_pos,
                                      const MeshRenderOptions& opts,
                                      bool use_fbo) {
        if (!initialized_)
            return std::unexpected("MeshRenderer not initialized");

        if (mesh.vertex_count() == 0 || mesh.face_count() == 0)
            return {};

        auto upload_result = uploadMeshData(mesh);
        if (!upload_result)
            return upload_result;

        if (use_fbo) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_.get());
            glViewport(0, 0, fbo_width_, fbo_height_);

            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            const float far_depth = 1e10f;
            glClearBufferfv(GL_COLOR, 1, &far_depth);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        if (opts.backface_culling) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        } else {
            glDisable(GL_CULL_FACE);
        }

        // PBR pass
        {
            ShaderScope scope(pbr_shader_);

            glm::mat3 normal_matrix = glm::inverseTranspose(glm::mat3(model));

            pbr_shader_->set_uniform("u_model", model);
            pbr_shader_->set_uniform("u_view", view);
            pbr_shader_->set_uniform("u_projection", projection);
            pbr_shader_->set_uniform("u_normal_matrix", normal_matrix);
            pbr_shader_->set_uniform("u_camera_pos", camera_pos);
            pbr_shader_->set_uniform("u_light_dir", glm::normalize(opts.light_dir));
            pbr_shader_->set_uniform("u_light_intensity", opts.light_intensity);
            pbr_shader_->set_uniform("u_ambient", opts.ambient);

            const auto& mat = mesh.materials.empty()
                                  ? lfs::core::Material{}
                                  : mesh.materials[0];

            pbr_shader_->set_uniform("u_base_color", glm::vec4(mat.base_color));
            pbr_shader_->set_uniform("u_metallic", mat.metallic);
            pbr_shader_->set_uniform("u_roughness", mat.roughness);
            pbr_shader_->set_uniform("u_emissive", glm::vec3(mat.emissive));

            pbr_shader_->set_uniform("u_has_albedo_tex", false);
            pbr_shader_->set_uniform("u_has_normal_tex", false);
            pbr_shader_->set_uniform("u_has_metallic_roughness_tex", false);
            pbr_shader_->set_uniform("u_has_vertex_colors", mesh.has_colors());

            glBindVertexArray(vao_.get());
            glDrawElements(GL_TRIANGLES,
                           static_cast<GLsizei>(mesh.face_count() * 3),
                           GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);
        }

        // Wireframe overlay
        if (opts.wireframe_overlay) {
            ShaderScope scope(wireframe_shader_);

            glm::mat4 mvp = projection * view * model;
            wireframe_shader_->set_uniform("u_mvp", mvp);
            wireframe_shader_->set_uniform("u_color", opts.wireframe_color);

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glLineWidth(opts.wireframe_width);
            glEnable(GL_POLYGON_OFFSET_LINE);
            glPolygonOffset(-1.0f, -1.0f);

            glBindVertexArray(vao_.get());
            glDrawElements(GL_TRIANGLES,
                           static_cast<GLsizei>(mesh.face_count() * 3),
                           GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_POLYGON_OFFSET_LINE);
            glLineWidth(1.0f);
        }

        glDisable(GL_CULL_FACE);

        if (use_fbo) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        return {};
    }

} // namespace lfs::rendering

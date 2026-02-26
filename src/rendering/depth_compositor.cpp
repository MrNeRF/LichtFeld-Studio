/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "depth_compositor.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include <glad/glad.h>

namespace lfs::rendering {

    Result<void> DepthCompositor::initialize() {
        if (initialized_)
            return {};

        auto shader_result = load_shader("depth_composite",
                                         "depth_composite.vert",
                                         "depth_composite.frag", false);
        if (!shader_result) {
            LOG_ERROR("Failed to load depth composite shader: {}", shader_result.error().what());
            return std::unexpected(shader_result.error().what());
        }
        shader_ = std::move(*shader_result);

        // clang-format off
        constexpr float QUAD_VERTICES[] = {
            -1.0f, -1.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 1.0f,
             1.0f,  1.0f, 1.0f, 1.0f,
        };
        // clang-format on

        auto vao_result = create_vao();
        if (!vao_result)
            return std::unexpected(vao_result.error());
        vao_ = std::move(*vao_result);

        auto vbo_result = create_vbo();
        if (!vbo_result)
            return std::unexpected(vbo_result.error());
        vbo_ = std::move(*vbo_result);

        glBindVertexArray(vao_.get());

        glBindBuffer(GL_ARRAY_BUFFER, vbo_.get());
        glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD_VERTICES), QUAD_VERTICES, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              reinterpret_cast<const void*>(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        initialized_ = true;
        return {};
    }

    Result<void> DepthCompositor::composite(GLuint splat_color_tex, GLuint splat_depth_tex,
                                            GLuint mesh_color_tex, GLuint mesh_depth_tex,
                                            float near_plane, float far_plane,
                                            bool flip_splat_y,
                                            const glm::vec2& splat_texcoord_scale,
                                            bool splat_depth_is_ndc) {
        if (!initialized_)
            return std::unexpected("DepthCompositor not initialized");

        GLStateGuard state_guard;
        ShaderScope scope(shader_);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, splat_color_tex);
        shader_->set_uniform("u_splat_color", 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, splat_depth_tex);
        shader_->set_uniform("u_splat_depth", 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, mesh_color_tex);
        shader_->set_uniform("u_mesh_color", 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, mesh_depth_tex);
        shader_->set_uniform("u_mesh_depth", 3);

        shader_->set_uniform("u_near_plane", near_plane);
        shader_->set_uniform("u_far_plane", far_plane);
        shader_->set_uniform("u_flip_splat_y", flip_splat_y);
        shader_->set_uniform("u_splat_texcoord_scale", splat_texcoord_scale);
        shader_->set_uniform("u_splat_depth_is_ndc", splat_depth_is_ndc);
        shader_->set_uniform("u_mesh_only", false);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_ALWAYS);

        glBindVertexArray(vao_.get());
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        return {};
    }

    Result<void> DepthCompositor::presentMeshOnly(GLuint mesh_color_tex, GLuint mesh_depth_tex) {
        if (!initialized_)
            return std::unexpected("DepthCompositor not initialized");

        GLStateGuard state_guard;
        ShaderScope scope(shader_);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, mesh_color_tex);
        shader_->set_uniform("u_mesh_color", 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, mesh_depth_tex);
        shader_->set_uniform("u_mesh_depth", 3);

        shader_->set_uniform("u_mesh_only", true);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_ALWAYS);

        glBindVertexArray(vao_.get());
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        return {};
    }

} // namespace lfs::rendering

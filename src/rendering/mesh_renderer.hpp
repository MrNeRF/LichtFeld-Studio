/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "rendering/rendering.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>

namespace lfs::core {
    struct MeshData;
}

namespace lfs::rendering {

    class MeshRenderer {
    public:
        Result<void> initialize();
        bool isInitialized() const { return initialized_; }

        Result<void> render(const lfs::core::MeshData& mesh,
                            const glm::mat4& model,
                            const glm::mat4& view,
                            const glm::mat4& projection,
                            const glm::vec3& camera_pos,
                            const MeshRenderOptions& opts,
                            bool use_fbo = true);

        GLuint getColorTexture() const { return color_texture_.get(); }
        GLuint getDepthTexture() const { return depth_texture_.get(); }
        int getWidth() const { return fbo_width_; }
        int getHeight() const { return fbo_height_; }

        void resize(int width, int height);

    private:
        Result<void> setupFBO(int width, int height);
        Result<void> uploadMeshData(const lfs::core::MeshData& mesh);

        ManagedShader pbr_shader_;
        ManagedShader wireframe_shader_;

        VAO vao_;
        VBO vbo_positions_;
        VBO vbo_normals_;
        VBO vbo_tangents_;
        VBO vbo_texcoords_;
        VBO vbo_colors_;
        EBO ebo_;

        FBO fbo_;
        Texture color_texture_;
        Texture depth_texture_;
        RBO depth_rbo_;

        int fbo_width_ = 0;
        int fbo_height_ = 0;
        int64_t uploaded_vertex_count_ = 0;
        int64_t uploaded_face_count_ = 0;

        bool initialized_ = false;
    };

} // namespace lfs::rendering

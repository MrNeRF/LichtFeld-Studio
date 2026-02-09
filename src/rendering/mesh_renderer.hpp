/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "rendering/rendering.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>

namespace lfs::core {
    struct MeshData;
    struct Material;
} // namespace lfs::core

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
                            bool use_fbo = false,
                            bool clear_fbo = true);

        GLuint getColorTexture() const { return color_texture_.get(); }
        GLuint getDepthTexture() const { return depth_texture_.get(); }
        GLuint getFramebuffer() const { return fbo_.get(); }
        GLuint getShadowDepthTexture() const { return shadow_depth_texture_.get(); }
        int getWidth() const { return fbo_width_; }
        int getHeight() const { return fbo_height_; }

        void resize(int width, int height);
        void blitToScreen(const glm::ivec2& dst_pos, const glm::ivec2& dst_size);

    private:
        Result<void> setupFBO(int width, int height);
        Result<void> setupShadowFBO(int resolution);
        Result<void> uploadMeshData(const lfs::core::MeshData& mesh);
        void uploadTextures(const lfs::core::MeshData& mesh);
        void bindMaterial(const lfs::core::Material& mat, size_t mat_idx, bool has_texcoords);

        ManagedShader pbr_shader_;
        ManagedShader wireframe_shader_;
        ManagedShader shadow_shader_;

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

        FBO shadow_fbo_;
        Texture shadow_depth_texture_;
        int shadow_map_resolution_ = 0;

        struct GLMaterialTextures {
            Texture albedo;
            Texture normal;
            Texture metallic_roughness;
        };
        std::unordered_map<size_t, GLMaterialTextures> material_textures_;
        uint32_t uploaded_texture_generation_ = 0;

        int fbo_width_ = 0;
        int fbo_height_ = 0;
        int64_t uploaded_vertex_count_ = 0;
        int64_t uploaded_face_count_ = 0;
        uint32_t uploaded_generation_ = 0;

        bool initialized_ = false;
    };

} // namespace lfs::rendering

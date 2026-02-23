/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <glad/glad.h>

namespace lfs::vis::gui {

    class RmlRenderInterface final : public Rml::RenderInterface {
    public:
        RmlRenderInterface();
        ~RmlRenderInterface() override;

        void SetViewport(int width, int height);
        void BeginFrame();
        void EndFrame();

        // -- Rml::RenderInterface --
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                                                    Rml::Span<const int> indices) override;
        void RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation,
                            Rml::TextureHandle texture) override;
        void ReleaseGeometry(Rml::CompiledGeometryHandle handle) override;

        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String& source) override;
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source_data,
                                           Rml::Vector2i dimensions) override;
        void ReleaseTexture(Rml::TextureHandle handle) override;

        void EnableScissorRegion(bool enable) override;
        void SetScissorRegion(Rml::Rectanglei region) override;

        void SetTransform(const Rml::Matrix4f* transform) override;

    private:
        bool initShaders();
        void destroyShaders();

        struct CompiledGeometry {
            GLuint vao = 0;
            GLuint vbo = 0;
            GLuint ebo = 0;
            int num_indices = 0;
        };

        GLuint program_color_ = 0;
        GLuint program_texture_ = 0;
        GLint u_transform_color_ = -1;
        GLint u_transform_texture_ = -1;
        GLint u_texture_ = -1;

        Rml::Matrix4f projection_;
        Rml::Matrix4f transform_;
        bool transform_enabled_ = false;

        int viewport_width_ = 0;
        int viewport_height_ = 0;

        bool shaders_valid_ = false;

        struct GLStateBackup {
            GLboolean blend;
            GLboolean cull_face;
            GLboolean depth_test;
            GLboolean scissor_test;
            GLboolean stencil_test;
            GLint blend_src_rgb, blend_dst_rgb;
            GLint blend_src_alpha, blend_dst_alpha;
            GLint viewport[4];
            GLint scissor[4];
            GLint active_texture;
            GLint program;
            GLint vao;
            GLint array_buffer;
        };
        GLStateBackup gl_backup_{};
    };

} // namespace lfs::vis::gui

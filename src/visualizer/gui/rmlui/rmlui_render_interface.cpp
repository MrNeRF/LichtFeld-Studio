/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rmlui/rmlui_render_interface.hpp"
#include "core/logger.hpp"

#include <RmlUi/Core/Core.h>
#include <cassert>
#include <cstring>
#include <stb_image.h>

namespace lfs::vis::gui {

    namespace {
        const char* VERT_SHADER = R"glsl(
#version 430 core
uniform mat4 u_transform;
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;
layout(location = 2) in vec2 a_texcoord;
out vec4 v_color;
out vec2 v_texcoord;
void main() {
    v_color = a_color;
    v_texcoord = a_texcoord;
    gl_Position = u_transform * vec4(a_position, 0.0, 1.0);
}
)glsl";

        const char* FRAG_COLOR = R"glsl(
#version 430 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
)glsl";

        const char* FRAG_TEXTURE = R"glsl(
#version 430 core
uniform sampler2D u_texture;
in vec4 v_color;
in vec2 v_texcoord;
out vec4 frag_color;
void main() {
    vec4 tex = texture(u_texture, v_texcoord);
    frag_color = vec4(tex.rgb * tex.a, tex.a) * v_color;
}
)glsl";

        GLuint compileShader(GLenum type, const char* source) {
            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &source, nullptr);
            glCompileShader(shader);

            GLint status = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
            if (!status) {
                char log[512];
                glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
                LOG_ERROR("RmlUI shader compile error: {}", log);
                glDeleteShader(shader);
                return 0;
            }
            return shader;
        }

        GLuint linkProgram(GLuint vert, GLuint frag) {
            GLuint program = glCreateProgram();
            glAttachShader(program, vert);
            glAttachShader(program, frag);
            glLinkProgram(program);

            GLint status = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &status);
            if (!status) {
                char log[512];
                glGetProgramInfoLog(program, sizeof(log), nullptr, log);
                LOG_ERROR("RmlUI program link error: {}", log);
                glDeleteProgram(program);
                return 0;
            }
            return program;
        }
    } // namespace

    RmlRenderInterface::RmlRenderInterface() {
        transform_ = Rml::Matrix4f::Identity();
        projection_ = Rml::Matrix4f::Identity();
        shaders_valid_ = initShaders();
        if (!shaders_valid_)
            LOG_ERROR("RmlUI render interface failed to initialize shaders");
    }

    RmlRenderInterface::~RmlRenderInterface() { destroyShaders(); }

    bool RmlRenderInterface::initShaders() {
        GLuint vs = compileShader(GL_VERTEX_SHADER, VERT_SHADER);
        if (!vs)
            return false;

        GLuint fs_color = compileShader(GL_FRAGMENT_SHADER, FRAG_COLOR);
        GLuint fs_texture = compileShader(GL_FRAGMENT_SHADER, FRAG_TEXTURE);

        if (!fs_color || !fs_texture) {
            glDeleteShader(vs);
            if (fs_color)
                glDeleteShader(fs_color);
            if (fs_texture)
                glDeleteShader(fs_texture);
            return false;
        }

        program_color_ = linkProgram(vs, fs_color);
        program_texture_ = linkProgram(vs, fs_texture);

        glDeleteShader(vs);
        glDeleteShader(fs_color);
        glDeleteShader(fs_texture);

        if (!program_color_ || !program_texture_) {
            destroyShaders();
            return false;
        }

        u_transform_color_ = glGetUniformLocation(program_color_, "u_transform");
        u_transform_texture_ = glGetUniformLocation(program_texture_, "u_transform");
        u_texture_ = glGetUniformLocation(program_texture_, "u_texture");

        return true;
    }

    void RmlRenderInterface::destroyShaders() {
        if (program_color_) {
            glDeleteProgram(program_color_);
            program_color_ = 0;
        }
        if (program_texture_) {
            glDeleteProgram(program_texture_);
            program_texture_ = 0;
        }
    }

    void RmlRenderInterface::SetViewport(int width, int height) {
        viewport_width_ = width;
        viewport_height_ = height;

        // Orthographic projection: top-left origin, Y down (RmlUI convention)
        projection_ = Rml::Matrix4f::Identity();
        projection_[0][0] = 2.0f / static_cast<float>(width);
        projection_[1][1] = -2.0f / static_cast<float>(height);
        projection_[3][0] = -1.0f;
        projection_[3][1] = 1.0f;
    }

    void RmlRenderInterface::BeginFrame() {
        assert(shaders_valid_);

        // Save GL state
        gl_backup_.blend = glIsEnabled(GL_BLEND);
        gl_backup_.cull_face = glIsEnabled(GL_CULL_FACE);
        gl_backup_.depth_test = glIsEnabled(GL_DEPTH_TEST);
        gl_backup_.scissor_test = glIsEnabled(GL_SCISSOR_TEST);
        gl_backup_.stencil_test = glIsEnabled(GL_STENCIL_TEST);
        glGetIntegerv(GL_BLEND_SRC_RGB, &gl_backup_.blend_src_rgb);
        glGetIntegerv(GL_BLEND_DST_RGB, &gl_backup_.blend_dst_rgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &gl_backup_.blend_src_alpha);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &gl_backup_.blend_dst_alpha);
        glGetIntegerv(GL_VIEWPORT, gl_backup_.viewport);
        glGetIntegerv(GL_SCISSOR_BOX, gl_backup_.scissor);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &gl_backup_.active_texture);
        glGetIntegerv(GL_CURRENT_PROGRAM, &gl_backup_.program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &gl_backup_.vao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &gl_backup_.array_buffer);

        // Set RmlUI rendering state
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA); // premultiplied alpha
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_STENCIL_TEST);
        glActiveTexture(GL_TEXTURE0);

        glViewport(0, 0, viewport_width_, viewport_height_);

        transform_enabled_ = false;
        transform_ = Rml::Matrix4f::Identity();
    }

    void RmlRenderInterface::EndFrame() {
        // Restore GL state
        if (gl_backup_.blend)
            glEnable(GL_BLEND);
        else
            glDisable(GL_BLEND);
        if (gl_backup_.cull_face)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);
        if (gl_backup_.depth_test)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);
        if (gl_backup_.scissor_test)
            glEnable(GL_SCISSOR_TEST);
        else
            glDisable(GL_SCISSOR_TEST);
        if (gl_backup_.stencil_test)
            glEnable(GL_STENCIL_TEST);
        else
            glDisable(GL_STENCIL_TEST);

        glBlendFuncSeparate(gl_backup_.blend_src_rgb, gl_backup_.blend_dst_rgb,
                            gl_backup_.blend_src_alpha, gl_backup_.blend_dst_alpha);
        glViewport(gl_backup_.viewport[0], gl_backup_.viewport[1], gl_backup_.viewport[2],
                   gl_backup_.viewport[3]);
        glScissor(gl_backup_.scissor[0], gl_backup_.scissor[1], gl_backup_.scissor[2],
                  gl_backup_.scissor[3]);
        glActiveTexture(gl_backup_.active_texture);
        glUseProgram(gl_backup_.program);
        glBindVertexArray(gl_backup_.vao);
        glBindBuffer(GL_ARRAY_BUFFER, gl_backup_.array_buffer);
    }

    Rml::CompiledGeometryHandle
    RmlRenderInterface::CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                                        Rml::Span<const int> indices) {
        auto* geom = new CompiledGeometry();
        geom->num_indices = static_cast<int>(indices.size());

        glGenVertexArrays(1, &geom->vao);
        glGenBuffers(1, &geom->vbo);
        glGenBuffers(1, &geom->ebo);

        glBindVertexArray(geom->vao);

        glBindBuffer(GL_ARRAY_BUFFER, geom->vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(vertices.size() * sizeof(Rml::Vertex)),
                     vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geom->ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(indices.size() * sizeof(int)), indices.data(),
                     GL_STATIC_DRAW);

        // Rml::Vertex layout: position(2f), colour(4ub), tex_coord(2f)
        static_assert(sizeof(Rml::Vertex) == 20, "Unexpected Rml::Vertex size");

        // position
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Rml::Vertex),
                              reinterpret_cast<const void*>(offsetof(Rml::Vertex, position)));
        // colour
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Rml::Vertex),
                              reinterpret_cast<const void*>(offsetof(Rml::Vertex, colour)));
        // tex_coord
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Rml::Vertex),
                              reinterpret_cast<const void*>(offsetof(Rml::Vertex, tex_coord)));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        return reinterpret_cast<Rml::CompiledGeometryHandle>(geom);
    }

    void RmlRenderInterface::RenderGeometry(Rml::CompiledGeometryHandle handle,
                                            Rml::Vector2f translation,
                                            Rml::TextureHandle texture) {
        auto* geom = reinterpret_cast<CompiledGeometry*>(handle);
        assert(geom);

        Rml::Matrix4f mvp = projection_;
        if (transform_enabled_)
            mvp *= transform_;

        // Apply translation
        mvp[3][0] += mvp[0][0] * translation.x + mvp[1][0] * translation.y;
        mvp[3][1] += mvp[0][1] * translation.x + mvp[1][1] * translation.y;

        if (texture) {
            glUseProgram(program_texture_);
            glUniformMatrix4fv(u_transform_texture_, 1, GL_FALSE, mvp.data());
            glUniform1i(u_texture_, 0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture));
        } else {
            glUseProgram(program_color_);
            glUniformMatrix4fv(u_transform_color_, 1, GL_FALSE, mvp.data());
        }

        glBindVertexArray(geom->vao);
        glDrawElements(GL_TRIANGLES, geom->num_indices, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);

        if (texture)
            glBindTexture(GL_TEXTURE_2D, 0);
    }

    void RmlRenderInterface::ReleaseGeometry(Rml::CompiledGeometryHandle handle) {
        auto* geom = reinterpret_cast<CompiledGeometry*>(handle);
        if (!geom)
            return;

        if (geom->vao)
            glDeleteVertexArrays(1, &geom->vao);
        if (geom->vbo)
            glDeleteBuffers(1, &geom->vbo);
        if (geom->ebo)
            glDeleteBuffers(1, &geom->ebo);
        delete geom;
    }

    Rml::TextureHandle RmlRenderInterface::LoadTexture(Rml::Vector2i& dimensions,
                                                       const Rml::String& source) {
        int w = 0, h = 0, channels = 0;
        unsigned char* data = stbi_load(source.c_str(), &w, &h, &channels, 4);
        if (!data) {
            LOG_WARN("RmlUI LoadTexture failed: {}", source);
            return 0;
        }

        dimensions.x = w;
        dimensions.y = h;

        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glBindTexture(GL_TEXTURE_2D, 0);

        stbi_image_free(data);
        return static_cast<Rml::TextureHandle>(tex);
    }

    Rml::TextureHandle RmlRenderInterface::GenerateTexture(Rml::Span<const Rml::byte> source_data,
                                                           Rml::Vector2i dimensions) {
        GLuint tex = 0;
        glGenTextures(1, &tex);
        if (!tex)
            return 0;

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, dimensions.x, dimensions.y, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, source_data.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        return static_cast<Rml::TextureHandle>(tex);
    }

    void RmlRenderInterface::ReleaseTexture(Rml::TextureHandle handle) {
        GLuint tex = static_cast<GLuint>(handle);
        if (tex)
            glDeleteTextures(1, &tex);
    }

    void RmlRenderInterface::EnableScissorRegion(bool enable) {
        if (enable)
            glEnable(GL_SCISSOR_TEST);
        else
            glDisable(GL_SCISSOR_TEST);
    }

    void RmlRenderInterface::SetScissorRegion(Rml::Rectanglei region) {
        // RmlUI uses top-left origin; GL scissor uses bottom-left
        int y_flipped = viewport_height_ - (region.Top() + region.Height());
        glScissor(region.Left(), y_flipped, region.Width(), region.Height());
    }

    void RmlRenderInterface::SetTransform(const Rml::Matrix4f* transform) {
        if (transform) {
            transform_enabled_ = true;
            transform_ = *transform;
        } else {
            transform_enabled_ = false;
            transform_ = Rml::Matrix4f::Identity();
        }
    }

} // namespace lfs::vis::gui

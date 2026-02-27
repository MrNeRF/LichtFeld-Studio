/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/gl_line_renderer.hpp"

#include <cassert>
#include <cmath>

namespace lfs::vis::gui {

    GLuint GLLineRenderer::program_ = 0;
    GLuint GLLineRenderer::vao_ = 0;
    GLuint GLLineRenderer::vbo_ = 0;

    void GLLineRenderer::ensureProgram() {
        if (program_)
            return;

        static const char* vs_src = R"(
            #version 330 core
            layout(location=0) in vec2 aPos;
            layout(location=1) in vec4 aColor;
            uniform vec2 uScreenSize;
            out vec4 vColor;
            void main() {
                vec2 ndc = vec2(
                    2.0 * aPos.x / uScreenSize.x - 1.0,
                    1.0 - 2.0 * aPos.y / uScreenSize.y
                );
                gl_Position = vec4(ndc, 0.0, 1.0);
                vColor = aColor;
            }
        )";
        static const char* fs_src = R"(
            #version 330 core
            in vec4 vColor;
            out vec4 fragColor;
            void main() {
                fragColor = vColor;
            }
        )";

        auto compile = [](GLenum type, const char* src) -> GLuint {
            GLuint s = glCreateShader(type);
            glShaderSource(s, 1, &src, nullptr);
            glCompileShader(s);
            return s;
        };

        GLuint vs = compile(GL_VERTEX_SHADER, vs_src);
        GLuint fs = compile(GL_FRAGMENT_SHADER, fs_src);

        program_ = glCreateProgram();
        glAttachShader(program_, vs);
        glAttachShader(program_, fs);
        glLinkProgram(program_);
        glDeleteShader(vs);
        glDeleteShader(fs);

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              reinterpret_cast<void*>(2 * sizeof(float)));
        glBindVertexArray(0);
    }

    void GLLineRenderer::begin(int screen_w, int screen_h) {
        assert(screen_w > 0 && screen_h > 0);
        screen_w_ = screen_w;
        screen_h_ = screen_h;
        vertices_.clear();
    }

    void GLLineRenderer::addLine(glm::vec2 p0, glm::vec2 p1, glm::vec4 color, float thickness) {
        if (thickness <= 1.0f) {
            vertices_.push_back({p0.x, p0.y, color.r, color.g, color.b, color.a});
            vertices_.push_back({p1.x, p1.y, color.r, color.g, color.b, color.a});

            const glm::vec2 mid = (p0 + p1) * 0.5f;
            vertices_.push_back({mid.x, mid.y, color.r, color.g, color.b, color.a});
            vertices_.push_back({p0.x, p0.y, color.r, color.g, color.b, color.a});
            vertices_.push_back({p1.x, p1.y, color.r, color.g, color.b, color.a});
            vertices_.push_back({mid.x, mid.y, color.r, color.g, color.b, color.a});
            return;
        }

        const glm::vec2 dir = p1 - p0;
        const float len = glm::length(dir);
        if (len < 0.001f)
            return;

        const glm::vec2 norm = glm::vec2(-dir.y, dir.x) / len;
        const float half = thickness * 0.5f;

        const glm::vec2 a = p0 + norm * half;
        const glm::vec2 b = p0 - norm * half;
        const glm::vec2 c = p1 - norm * half;
        const glm::vec2 d = p1 + norm * half;

        vertices_.push_back({a.x, a.y, color.r, color.g, color.b, color.a});
        vertices_.push_back({b.x, b.y, color.r, color.g, color.b, color.a});
        vertices_.push_back({c.x, c.y, color.r, color.g, color.b, color.a});
        vertices_.push_back({a.x, a.y, color.r, color.g, color.b, color.a});
        vertices_.push_back({c.x, c.y, color.r, color.g, color.b, color.a});
        vertices_.push_back({d.x, d.y, color.r, color.g, color.b, color.a});
    }

    void GLLineRenderer::addTriangleFilled(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec4 color) {
        vertices_.push_back({p0.x, p0.y, color.r, color.g, color.b, color.a});
        vertices_.push_back({p1.x, p1.y, color.r, color.g, color.b, color.a});
        vertices_.push_back({p2.x, p2.y, color.r, color.g, color.b, color.a});
    }

    void GLLineRenderer::addCircleFilled(glm::vec2 center, float radius, glm::vec4 color, int segments) {
        assert(segments >= 3);
        for (int i = 0; i < segments; ++i) {
            const float a0 = 2.0f * 3.14159265f * static_cast<float>(i) / static_cast<float>(segments);
            const float a1 = 2.0f * 3.14159265f * static_cast<float>(i + 1) / static_cast<float>(segments);
            addTriangleFilled(center,
                              center + glm::vec2(std::cos(a0), std::sin(a0)) * radius,
                              center + glm::vec2(std::cos(a1), std::sin(a1)) * radius,
                              color);
        }
    }

    void GLLineRenderer::end() {
        if (vertices_.empty())
            return;

        ensureProgram();

        GLint prev_program = 0;
        GLint prev_vao = 0;
        GLint prev_blend = 0;
        GLint prev_depth = 0;
        GLint prev_scissor = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &prev_program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_vao);
        glGetIntegerv(GL_BLEND, &prev_blend);
        glGetIntegerv(GL_DEPTH_TEST, &prev_depth);
        glGetIntegerv(GL_SCISSOR_TEST, &prev_scissor);

        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_SCISSOR_TEST);

        glUseProgram(program_);
        glUniform2f(glGetUniformLocation(program_, "uScreenSize"),
                    static_cast<float>(screen_w_), static_cast<float>(screen_h_));

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(vertices_.size() * sizeof(Vertex)),
                     vertices_.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices_.size()));

        glBindVertexArray(prev_vao);
        glUseProgram(prev_program);
        if (prev_blend)
            glEnable(GL_BLEND);
        else
            glDisable(GL_BLEND);
        if (prev_depth)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);
        if (prev_scissor)
            glEnable(GL_SCISSOR_TEST);
        else
            glDisable(GL_SCISSOR_TEST);
    }

    glm::vec4 GLLineRenderer::fromU32(uint32_t abgr) {
        return {static_cast<float>((abgr >> 0) & 0xFF) / 255.0f,
                static_cast<float>((abgr >> 8) & 0xFF) / 255.0f,
                static_cast<float>((abgr >> 16) & 0xFF) / 255.0f,
                static_cast<float>((abgr >> 24) & 0xFF) / 255.0f};
    }

    void GLLineRenderer::destroyGLResources() {
        if (program_) {
            glDeleteProgram(program_);
            program_ = 0;
        }
        if (vbo_) {
            glDeleteBuffers(1, &vbo_);
            vbo_ = 0;
        }
        if (vao_) {
            glDeleteVertexArrays(1, &vao_);
            vao_ = 0;
        }
    }

} // namespace lfs::vis::gui

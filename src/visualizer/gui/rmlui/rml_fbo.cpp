/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rmlui/rml_fbo.hpp"
#include "core/logger.hpp"

#include <cassert>
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        void setPremultipliedBlend(const ImDrawList*, const ImDrawCmd*) {
            glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA,
                                GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        }

        void restoreStandardBlend(const ImDrawList*, const ImDrawCmd*) {
            glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                                GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        }
    } // namespace

    RmlFBO::~RmlFBO() { destroy(); }

    void RmlFBO::ensure(int w, int h) {
        assert(w > 0 && h > 0);
        if (w <= 0 || h <= 0)
            return;
        if (fbo_ && width_ == w && height_ == h)
            return;

        destroy();
        width_ = w;
        height_ = h;

        glGenFramebuffers(1, &fbo_);
        glGenTextures(1, &texture_);
        glGenRenderbuffers(1, &depth_stencil_);

        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, depth_stencil_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
                                  depth_stencil_);

        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("RmlFBO: framebuffer incomplete (status=0x{:X}, {}x{}, tex={}, rbo={})",
                      status, w, h, texture_, depth_stencil_);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            destroy();
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void RmlFBO::bind(GLint* prev_fbo) {
        assert(fbo_);
        assert(prev_fbo);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, prev_fbo);

        const GLboolean had_scissor = glIsEnabled(GL_SCISSOR_TEST);
        GLint prev_scissor[4] = {};
        glGetIntegerv(GL_SCISSOR_BOX, prev_scissor);
        if (had_scissor)
            glDisable(GL_SCISSOR_TEST);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        if (had_scissor) {
            glEnable(GL_SCISSOR_TEST);
            glScissor(prev_scissor[0], prev_scissor[1], prev_scissor[2], prev_scissor[3]);
        }
    }

    void RmlFBO::unbind(GLint prev_fbo) {
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
    }

    void RmlFBO::blitToDrawList(ImDrawList* dl, ImVec2 pos, ImVec2 size) const {
        assert(dl);
        assert(texture_);
        dl->AddCallback(setPremultipliedBlend, nullptr);
        const ImVec2 p1 = {pos.x + size.x, pos.y + size.y};
        dl->AddImage(static_cast<ImTextureID>(static_cast<uintptr_t>(texture_)),
                     pos, p1, {0, 1}, {1, 0});
        dl->AddCallback(restoreStandardBlend, nullptr);
    }

    void RmlFBO::blitAsImage(float w, float h) {
        assert(texture_);
        auto* dl = ImGui::GetWindowDrawList();
        dl->AddCallback(setPremultipliedBlend, nullptr);
        ImGui::Image(static_cast<ImTextureID>(static_cast<uintptr_t>(texture_)),
                     ImVec2(w, h), ImVec2(0, 1), ImVec2(1, 0));
        dl->AddCallback(restoreStandardBlend, nullptr);
    }

    GLuint RmlFBO::blit_program_ = 0;
    GLuint RmlFBO::blit_vao_ = 0;
    GLuint RmlFBO::blit_vbo_ = 0;

    void RmlFBO::ensureBlitProgram() {
        if (blit_program_)
            return;

        static const char* vs_src = R"(
            #version 330 core
            layout(location=0) in vec2 aPos;
            layout(location=1) in vec2 aUV;
            out vec2 vUV;
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                vUV = aUV;
            }
        )";
        static const char* fs_src = R"(
            #version 330 core
            in vec2 vUV;
            out vec4 fragColor;
            uniform sampler2D uTex;
            void main() {
                fragColor = texture(uTex, vUV);
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

        blit_program_ = glCreateProgram();
        glAttachShader(blit_program_, vs);
        glAttachShader(blit_program_, fs);
        glLinkProgram(blit_program_);
        glDeleteShader(vs);
        glDeleteShader(fs);

        glGenVertexArrays(1, &blit_vao_);
        glGenBuffers(1, &blit_vbo_);
        glBindVertexArray(blit_vao_);
        glBindBuffer(GL_ARRAY_BUFFER, blit_vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              reinterpret_cast<void*>(2 * sizeof(float)));
        glBindVertexArray(0);
    }

    void RmlFBO::blitToScreen(float x, float y, float w, float h, int screen_w, int screen_h) const {
        assert(texture_);
        assert(screen_w > 0 && screen_h > 0);

        ensureBlitProgram();

        static constexpr float verts[] = {
            -1.0f,
            -1.0f,
            0.0f,
            0.0f,
            1.0f,
            -1.0f,
            1.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            1.0f,
            -1.0f,
            1.0f,
            0.0f,
            1.0f,
        };

        GLint prev_program = 0;
        GLint prev_vao = 0;
        GLint prev_blend = 0;
        GLint prev_depth = 0;
        GLint prev_scissor = 0;
        GLint prev_viewport[4] = {};
        glGetIntegerv(GL_CURRENT_PROGRAM, &prev_program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_vao);
        glGetIntegerv(GL_BLEND, &prev_blend);
        glGetIntegerv(GL_DEPTH_TEST, &prev_depth);
        glGetIntegerv(GL_SCISSOR_TEST, &prev_scissor);
        glGetIntegerv(GL_VIEWPORT, prev_viewport);

        glViewport(static_cast<GLint>(x),
                   screen_h - static_cast<GLint>(y + h),
                   static_cast<GLsizei>(w),
                   static_cast<GLsizei>(h));

        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_SCISSOR_TEST);

        glUseProgram(blit_program_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glUniform1i(glGetUniformLocation(blit_program_, "uTex"), 0);

        glBindVertexArray(blit_vao_);
        glBindBuffer(GL_ARRAY_BUFFER, blit_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);
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

        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }

    void RmlFBO::destroyBlitResources() {
        if (blit_program_) {
            glDeleteProgram(blit_program_);
            blit_program_ = 0;
        }
        if (blit_vbo_) {
            glDeleteBuffers(1, &blit_vbo_);
            blit_vbo_ = 0;
        }
        if (blit_vao_) {
            glDeleteVertexArrays(1, &blit_vao_);
            blit_vao_ = 0;
        }
    }

    void RmlFBO::destroy() {
        if (texture_) {
            glDeleteTextures(1, &texture_);
            texture_ = 0;
        }
        if (depth_stencil_) {
            glDeleteRenderbuffers(1, &depth_stencil_);
            depth_stencil_ = 0;
        }
        if (fbo_) {
            glDeleteFramebuffers(1, &fbo_);
            fbo_ = 0;
        }
        width_ = 0;
        height_ = 0;
    }

} // namespace lfs::vis::gui

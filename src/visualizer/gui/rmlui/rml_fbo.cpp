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

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("RmlFBO: framebuffer incomplete");
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
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    }

    void RmlFBO::unbind(GLint prev_fbo) {
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
    }

    void RmlFBO::blitToDrawList(ImDrawList* dl, ImVec2 pos, ImVec2 size) {
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

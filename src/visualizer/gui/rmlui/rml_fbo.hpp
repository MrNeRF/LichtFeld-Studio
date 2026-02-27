/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <glad/glad.h>
#include <imgui.h>

namespace lfs::vis::gui {

    class LFS_VIS_API RmlFBO {
    public:
        RmlFBO() = default;
        ~RmlFBO();

        RmlFBO(const RmlFBO&) = delete;
        RmlFBO& operator=(const RmlFBO&) = delete;

        void ensure(int w, int h);
        void bind(GLint* prev_fbo);
        void unbind(GLint prev_fbo);
        void blitToDrawList(ImDrawList* dl, ImVec2 pos, ImVec2 size) const;
        void blitAsImage(float w, float h);
        void blitToScreen(float x, float y, float w, float h, int screen_w, int screen_h) const;
        GLuint texture() const { return texture_; }
        int width() const { return width_; }
        int height() const { return height_; }
        bool valid() const { return fbo_ != 0; }
        void destroy();

        static void destroyBlitResources();

    private:
        static void ensureBlitProgram();

        static GLuint blit_program_;
        static GLuint blit_vao_;
        static GLuint blit_vbo_;

        GLuint fbo_ = 0;
        GLuint texture_ = 0;
        GLuint depth_stencil_ = 0;
        int width_ = 0;
        int height_ = 0;
    };

} // namespace lfs::vis::gui

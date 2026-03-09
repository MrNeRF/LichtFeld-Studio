/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <glad/glad.h>

namespace lfs::vis::gui {

    class LFS_VIS_API RmlFBO {
    public:
        RmlFBO() = default;
        ~RmlFBO();

        RmlFBO(const RmlFBO&) = delete;
        RmlFBO& operator=(const RmlFBO&) = delete;

        bool ensure(int w, int h);
        void bind(GLint* prev_fbo);
        void unbind(GLint prev_fbo);
        void blitAsImage(float w, float h);
        void blitToScreen(float x, float y, float w, float h, int screen_w, int screen_h) const;
        void blitToScreenClipped(float x, float y, float w, float h,
                                 int screen_w, int screen_h,
                                 float clip_x1, float clip_y1,
                                 float clip_x2, float clip_y2) const;
        GLuint fbo() const { return fbo_; }
        GLuint texture() const { return texture_; }
        int width() const { return width_; }
        int height() const { return height_; }
        bool valid() const { return fbo_ != 0; }
        void destroy();

        static void destroyBlitResources();

    private:
        static void ensureBlitProgram();
        void reallocate(int w, int h);
        float u_scale() const { return alloc_w_ > 0 ? static_cast<float>(width_) / static_cast<float>(alloc_w_) : 1.0f; }
        float v_scale() const { return alloc_h_ > 0 ? static_cast<float>(height_) / static_cast<float>(alloc_h_) : 1.0f; }

        static GLuint blit_program_;
        static GLuint blit_vao_;
        static GLuint blit_vbo_;

        GLuint fbo_ = 0;
        GLuint texture_ = 0;
        GLuint depth_stencil_ = 0;
        int width_ = 0;
        int height_ = 0;
        int alloc_w_ = 0;
        int alloc_h_ = 0;
    };

} // namespace lfs::vis::gui

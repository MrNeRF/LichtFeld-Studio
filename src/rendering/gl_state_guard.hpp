/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <concepts>
#include <expected>
#include <glad/glad.h>
#include <string>

namespace lfs::rendering {

    // Error handling with std::expected (C++23)
    template <typename T>
    using Result = std::expected<T, std::string>;

    // Concept for renderable objects (C++20)
    template <typename T>
    concept Renderable = requires(T t, glm::mat4 m) {
        { t.render(m, m) } -> std::same_as<void>;
        { t.isInitialized() } -> std::convertible_to<bool>;
    };

    // RAII class for OpenGL state management
    class GLStateGuard {
    public:
        struct State {
            GLint viewport[4];
            GLboolean depth_test;
            GLboolean blend;
            GLboolean stencil_test;
            GLboolean scissor_test;
            GLboolean cull_face;
            GLint depth_func;
            GLint blend_src_rgb;
            GLint blend_dst_rgb;
            GLint blend_src_alpha;
            GLint blend_dst_alpha;
            GLint blend_equation_rgb;
            GLint blend_equation_alpha;
            GLboolean depth_mask;
            GLboolean color_mask[4];
            GLint active_texture;
            GLint current_program;
            GLint vertex_array_binding;
            GLint texture_binding_2d;
            GLint texture_binding_2d_array;
            GLfloat line_width;
            GLboolean line_smooth;
            GLint unpack_alignment;
        };

    private:
        State saved_;
        bool restored_ = false;

    public:
        GLStateGuard() {
            LOG_TIMER_TRACE("GLStateGuard::save");

            // Save comprehensive OpenGL state
            glGetIntegerv(GL_VIEWPORT, saved_.viewport);
            saved_.depth_test = glIsEnabled(GL_DEPTH_TEST);
            saved_.blend = glIsEnabled(GL_BLEND);
            saved_.stencil_test = glIsEnabled(GL_STENCIL_TEST);
            saved_.scissor_test = glIsEnabled(GL_SCISSOR_TEST);
            saved_.cull_face = glIsEnabled(GL_CULL_FACE);
            glGetIntegerv(GL_DEPTH_FUNC, &saved_.depth_func);
            glGetIntegerv(GL_BLEND_SRC_RGB, &saved_.blend_src_rgb);
            glGetIntegerv(GL_BLEND_DST_RGB, &saved_.blend_dst_rgb);
            glGetIntegerv(GL_BLEND_SRC_ALPHA, &saved_.blend_src_alpha);
            glGetIntegerv(GL_BLEND_DST_ALPHA, &saved_.blend_dst_alpha);
            glGetIntegerv(GL_BLEND_EQUATION_RGB, &saved_.blend_equation_rgb);
            glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &saved_.blend_equation_alpha);
            glGetBooleanv(GL_DEPTH_WRITEMASK, &saved_.depth_mask);
            glGetBooleanv(GL_COLOR_WRITEMASK, saved_.color_mask);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &saved_.active_texture);
            glGetIntegerv(GL_CURRENT_PROGRAM, &saved_.current_program);
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &saved_.vertex_array_binding);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &saved_.texture_binding_2d);
            glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &saved_.texture_binding_2d_array);
            glGetFloatv(GL_LINE_WIDTH, &saved_.line_width);
            saved_.line_smooth = glIsEnabled(GL_LINE_SMOOTH);
            glGetIntegerv(GL_UNPACK_ALIGNMENT, &saved_.unpack_alignment);

            LOG_TRACE("GLStateGuard: Saved OpenGL state");
        }

        ~GLStateGuard() {
            if (!restored_) {
                restore();
            }
        }

        // Prevent copying
        GLStateGuard(const GLStateGuard&) = delete;
        GLStateGuard& operator=(const GLStateGuard&) = delete;

        // Allow moving
        GLStateGuard(GLStateGuard&& other) noexcept
            : saved_(other.saved_),
              restored_(other.restored_) {
            other.restored_ = true; // Prevent other from restoring
        }

        void restore() {
            if (restored_)
                return;

            LOG_TIMER_TRACE("GLStateGuard::restore");

            glViewport(saved_.viewport[0], saved_.viewport[1],
                       saved_.viewport[2], saved_.viewport[3]);

            if (saved_.depth_test)
                glEnable(GL_DEPTH_TEST);
            else
                glDisable(GL_DEPTH_TEST);

            if (saved_.blend)
                glEnable(GL_BLEND);
            else
                glDisable(GL_BLEND);

            if (saved_.stencil_test)
                glEnable(GL_STENCIL_TEST);
            else
                glDisable(GL_STENCIL_TEST);

            if (saved_.scissor_test)
                glEnable(GL_SCISSOR_TEST);
            else
                glDisable(GL_SCISSOR_TEST);

            if (saved_.cull_face)
                glEnable(GL_CULL_FACE);
            else
                glDisable(GL_CULL_FACE);

            glDepthFunc(saved_.depth_func);
            glBlendFuncSeparate(saved_.blend_src_rgb, saved_.blend_dst_rgb,
                                saved_.blend_src_alpha, saved_.blend_dst_alpha);
            glBlendEquationSeparate(saved_.blend_equation_rgb, saved_.blend_equation_alpha);
            glDepthMask(saved_.depth_mask);
            glColorMask(saved_.color_mask[0], saved_.color_mask[1],
                        saved_.color_mask[2], saved_.color_mask[3]);
            glActiveTexture(saved_.active_texture);
            glUseProgram(saved_.current_program);
            glBindVertexArray(saved_.vertex_array_binding);
            glBindTexture(GL_TEXTURE_2D, saved_.texture_binding_2d);
            glBindTexture(GL_TEXTURE_2D_ARRAY, saved_.texture_binding_2d_array);
            glLineWidth(saved_.line_width);

            if (saved_.line_smooth)
                glEnable(GL_LINE_SMOOTH);
            else
                glDisable(GL_LINE_SMOOTH);

            glPixelStorei(GL_UNPACK_ALIGNMENT, saved_.unpack_alignment);

            restored_ = true;
            LOG_TRACE("GLStateGuard: Restored OpenGL state");
        }

        // Get saved state for inspection
        const State& savedState() const { return saved_; }
    };

    // Scoped guard for specific state subsets
    class GLViewportGuard {
        GLint saved_viewport_[4];

    public:
        GLViewportGuard() {
            glGetIntegerv(GL_VIEWPORT, saved_viewport_);
            LOG_TRACE("GLViewportGuard: Saved viewport ({}, {}, {}, {})",
                      saved_viewport_[0], saved_viewport_[1], saved_viewport_[2], saved_viewport_[3]);
        }

        ~GLViewportGuard() {
            glViewport(saved_viewport_[0], saved_viewport_[1],
                       saved_viewport_[2], saved_viewport_[3]);
            LOG_TRACE("GLViewportGuard: Restored viewport");
        }
    };

    class GLLineGuard {
        GLfloat saved_width_;
        GLboolean saved_smooth_;

    public:
        GLLineGuard(float width) {
            glGetFloatv(GL_LINE_WIDTH, &saved_width_);
            saved_smooth_ = glIsEnabled(GL_LINE_SMOOTH);

            glEnable(GL_LINE_SMOOTH);
            glLineWidth(width);
            LOG_TRACE("GLLineGuard: Set line width to {} (was {})", width, saved_width_);
        }

        ~GLLineGuard() {
            glLineWidth(saved_width_);
            if (!saved_smooth_)
                glDisable(GL_LINE_SMOOTH);
            LOG_TRACE("GLLineGuard: Restored line width to {}", saved_width_);
        }
    };
} // namespace lfs::rendering
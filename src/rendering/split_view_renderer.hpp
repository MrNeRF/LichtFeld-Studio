/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "framebuffer.hpp"
#include "gl_resources.hpp"
#include "rendering/rendering.hpp"
#include "rendering_pipeline.hpp"
#include "shader_manager.hpp"
#include <memory>

namespace lfs::rendering {

    class SplitViewRenderer {
    public:
        SplitViewRenderer() = default;
        ~SplitViewRenderer() = default;

        Result<void> initialize();

        Result<RenderResult> render(
            const SplitViewRequest& request,
            RenderingPipeline& pipeline,
            ScreenQuadRenderer& screen_renderer,
            ManagedShader& quad_shader);

    private:
        // Framebuffers for each panel
        std::unique_ptr<FrameBuffer> left_framebuffer_;
        std::unique_ptr<FrameBuffer> right_framebuffer_;

        // Split view compositing shader
        ManagedShader split_shader_;

        // Simple texture blit shader for GT images
        ManagedShader texture_blit_shader_;

        // Quad VAO for rendering
        VAO quad_vao_;
        VBO quad_vbo_;

        bool initialized_ = false;

        Result<void> createFramebuffers(int width, int height);
        Result<void> setupQuad();
        Result<void> compositeSplitView(
            GLuint left_texture,
            GLuint right_texture,
            float split_position,
            const glm::vec2& left_texcoord_scale,
            const glm::vec2& right_texcoord_scale,
            const glm::vec4& divider_color,
            int viewport_width);

        // New method for rendering different content types
        Result<void> renderPanelContent(
            FrameBuffer* framebuffer,
            const SplitViewPanel& panel,
            const SplitViewRequest& request,
            RenderingPipeline& pipeline,
            ScreenQuadRenderer& screen_renderer,
            ManagedShader& quad_shader);

        // Helper to blit a texture to current framebuffer
        Result<void> blitTextureToFramebuffer(GLuint texture_id);
    };

} // namespace lfs::rendering
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
#include <optional>

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
        std::unique_ptr<FrameBuffer> left_framebuffer_;
        std::unique_ptr<FrameBuffer> right_framebuffer_;

        ManagedShader split_shader_;
        ManagedShader panel_shader_;
        ManagedShader texture_blit_shader_;

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
            int viewport_width,
            bool flip_left_y,
            bool flip_right_y);

        Result<std::optional<RenderingPipeline::RenderResult>> renderPanelContent(
            FrameBuffer* framebuffer,
            const SplitViewPanel& panel,
            const SplitViewRequest& request,
            RenderingPipeline& pipeline,
            ScreenQuadRenderer& screen_renderer,
            ManagedShader& quad_shader);

        Result<void> blitTextureToFramebuffer(GLuint texture_id);
    };

} // namespace lfs::rendering

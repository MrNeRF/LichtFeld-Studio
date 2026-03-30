/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "framebuffer.hpp"
#include "gl_resources.hpp"
#include "render_target_pool.hpp"
#include "rendering/rendering.hpp"
#include "rendering_pipeline.hpp"
#include "screen_renderer.hpp"
#include "shader_manager.hpp"
#include <array>
#include <memory>
#include <optional>

namespace lfs::rendering {

    class SplitViewRenderer {
    public:
        SplitViewRenderer() = default;
        ~SplitViewRenderer() = default;

        Result<void> initialize();

        Result<SplitViewFrameResult> renderGpuFrame(
            const SplitViewRequest& request,
            RenderTargetPool& render_target_pool,
            RenderingEngine& engine);

    private:
        struct SplitViewTargets {
            std::shared_ptr<FrameBuffer> composite;
        };

        struct PanelRenderOutput {
            GLuint texture_id = 0;
            glm::vec2 texcoord_scale{1.0f, 1.0f};
            std::optional<FrameMetadata> metadata;
            bool flip_y = false;
        };

        ManagedShader split_shader_;

        VAO quad_vao_;
        VBO quad_vbo_;
        std::array<std::unique_ptr<ScreenQuadRenderer>, 2> panel_upload_renderers_;

        bool initialized_ = false;

        Result<SplitViewTargets> acquireTargets(RenderTargetPool& render_target_pool,
                                                const glm::ivec2& size);
        Result<void> setupQuad();
        Result<ScreenQuadRenderer*> ensurePanelUploadRenderer(size_t panel_index);
        Result<void> compositeSplitView(
            GLuint left_texture,
            GLuint right_texture,
            float split_position,
            const glm::vec2& left_region,
            const glm::vec2& right_region,
            const glm::vec2& left_texcoord_scale,
            const glm::vec2& right_texcoord_scale,
            bool normalize_left_x,
            bool normalize_right_x,
            bool flip_left_y,
            bool flip_right_y);

        Result<PanelRenderOutput> renderPanelContent(
            size_t panel_index,
            const SplitViewPanel& panel,
            const glm::ivec2& panel_size,
            RenderingEngine& engine);
        Result<std::array<PanelRenderOutput, 2>> renderBatchedGaussianPanels(
            const SplitViewRequest& request,
            RenderingEngine& engine);
    };

} // namespace lfs::rendering

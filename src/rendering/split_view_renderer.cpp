/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_renderer.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "gl_state_guard.hpp"
#include <glad/glad.h>

namespace gs::rendering {

    Result<void> SplitViewRenderer::initialize() {
        if (initialized_) {
            return {};
        }

        LOG_DEBUG("Initializing SplitViewRenderer");

        // Load split view shader
        auto shader_result = load_shader("split_view", "split_view.vert", "split_view.frag", false);
        if (!shader_result) {
            LOG_ERROR("Failed to load split view shader: {}", shader_result.error().what());
            return std::unexpected("Failed to load split view shader");
        }
        split_shader_ = std::move(*shader_result);

        // Load texture blit shader for GT images
        auto blit_result = load_shader("texture_blit", "screen_quad.vert", "texture_blit.frag", false);
        if (!blit_result) {
            LOG_WARN("Failed to load texture blit shader, will use quad shader");
            // We can fallback to using the regular quad shader if needed
        } else {
            texture_blit_shader_ = std::move(*blit_result);
        }

        // Setup the quad for rendering
        if (auto result = setupQuad(); !result) {
            return result;
        }

        initialized_ = true;
        LOG_DEBUG("SplitViewRenderer initialized successfully");
        return {};
    }

    Result<void> SplitViewRenderer::setupQuad() {
        // Create VAO and VBO for full-screen quad
        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected("Failed to create VAO");
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected("Failed to create VBO");
        }
        quad_vbo_ = std::move(*vbo_result);

        // Full-screen quad vertices with texture coordinates
        float quad_vertices[] = {
            // positions   // texCoords
            -1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,

            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f};

        // Build VAO
        VAOBuilder builder(std::move(*vao_result));
        std::span<const float> vertices_span(quad_vertices, 24);

        builder.attachVBO(quad_vbo_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0, .size = 2, .type = GL_FLOAT, .stride = 4 * sizeof(float), .offset = nullptr})
            .setAttribute({.index = 1, .size = 2, .type = GL_FLOAT, .stride = 4 * sizeof(float), .offset = (void*)(2 * sizeof(float))});

        quad_vao_ = builder.build();
        LOG_DEBUG("Created full-screen quad VAO");
        return {};
    }

    Result<void> SplitViewRenderer::createFramebuffers(int width, int height) {
        // Always check and resize if dimensions don't match
        if (!left_framebuffer_) {
            left_framebuffer_ = std::make_unique<FrameBuffer>();
            LOG_DEBUG("Created left framebuffer");
        }

        if (!right_framebuffer_) {
            right_framebuffer_ = std::make_unique<FrameBuffer>();
            LOG_DEBUG("Created right framebuffer");
        }

        // Always resize if dimensions don't match current size
        if (left_framebuffer_->getWidth() != width || left_framebuffer_->getHeight() != height) {
            LOG_DEBUG("Resizing left framebuffer from {}x{} to {}x{}",
                      left_framebuffer_->getWidth(), left_framebuffer_->getHeight(), width, height);
            left_framebuffer_->resize(width, height);
        }

        if (right_framebuffer_->getWidth() != width || right_framebuffer_->getHeight() != height) {
            LOG_DEBUG("Resizing right framebuffer from {}x{} to {}x{}",
                      right_framebuffer_->getWidth(), right_framebuffer_->getHeight(), width, height);
            right_framebuffer_->resize(width, height);
        }

        return {};
    }

    Result<void> SplitViewRenderer::blitTextureToFramebuffer(GLuint texture_id) {
        LOG_TRACE("Blitting texture {} to current framebuffer", texture_id);

        // Disable depth test for 2D texture blit
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Use texture blit shader if available, otherwise fallback
        ManagedShader* shader = texture_blit_shader_.valid() ? &texture_blit_shader_ : nullptr;

        if (shader) {
            if (auto result = shader->bind(); !result) {
                LOG_ERROR("Failed to bind texture blit shader: {}", result.error());
                return result;
            }

            // Set texture
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            shader->set("texture0", 0);
        } else {
            // Simple OpenGL blit without shader
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_id);
        }

        // Draw fullscreen quad
        VAOBinder vao_bind(quad_vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        if (shader) {
            shader->unbind();
        }

        // Restore state
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

        return {};
    }

    Result<void> SplitViewRenderer::renderPanelContent(
        FrameBuffer* framebuffer,
        const SplitViewPanel& panel,
        const SplitViewRequest& request,
        RenderingPipeline& pipeline,
        ScreenQuadRenderer& screen_renderer,
        ManagedShader& quad_shader) {

        LOG_TRACE("Rendering panel '{}' with content type {}",
                  panel.label, static_cast<int>(panel.content_type));

        framebuffer->bind();

        // Set viewport to framebuffer size
        glViewport(0, 0, request.viewport.size.x, request.viewport.size.y);
        glClearColor(request.background_color.r, request.background_color.g,
                     request.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        switch (panel.content_type) {
        case PanelContentType::Model3D: {
            // Original path for 3D models
            if (!panel.model) {
                LOG_ERROR("Model3D panel has no model");
                framebuffer->unbind();
                return std::unexpected("Model3D panel has no model");
            }

            // Create render request
            RenderingPipeline::RenderRequest base_req{
                .view_rotation = request.viewport.rotation,
                .view_translation = request.viewport.translation,
                .viewport_size = request.viewport.size,
                .fov = request.viewport.fov,
                .scaling_modifier = request.scaling_modifier,
                .antialiasing = request.antialiasing,
                .render_mode = RenderMode::RGB,
                .crop_box = nullptr,
                .background_color = request.background_color,
                .point_cloud_mode = request.point_cloud_mode,
                .voxel_size = request.voxel_size,
                .gut = request.gut,
                .sh_degree = request.sh_degree};

            // Handle crop box if present
            std::unique_ptr<geometry::BoundingBox> temp_crop_box;
            if (request.crop_box.has_value()) {
                temp_crop_box = std::make_unique<geometry::BoundingBox>();
                temp_crop_box->setBounds(request.crop_box->min, request.crop_box->max);
                geometry::EuclideanTransform transform(request.crop_box->transform);
                temp_crop_box->setworld2BBox(transform);
                base_req.crop_box = temp_crop_box.get();
            }

            auto render_result = pipeline.render(*panel.model, base_req);
            if (!render_result) {
                LOG_ERROR("Failed to render model: {}", render_result.error());
                framebuffer->unbind();
                return std::unexpected(render_result.error());
            }

            // Present to framebuffer
            if (auto upload_result = RenderingPipeline::uploadToScreen(*render_result, screen_renderer, request.viewport.size);
                !upload_result) {
                LOG_ERROR("Failed to upload model: {}", upload_result.error());
            } else {
                glViewport(0, 0, request.viewport.size.x, request.viewport.size.y);
                if (auto render_result = screen_renderer.render(quad_shader); !render_result) {
                    LOG_ERROR("Failed to render to framebuffer: {}", render_result.error());
                }
            }
            break;
        }

        case PanelContentType::Image2D:
        case PanelContentType::CachedRender: {
            // New path for textures (GT images or cached renders)
            if (panel.texture_id == 0) {
                LOG_ERROR("Panel has invalid texture ID");
                framebuffer->unbind();
                return std::unexpected("Panel has invalid texture ID");
            }

            // Simply blit the texture to the framebuffer
            if (auto result = blitTextureToFramebuffer(panel.texture_id); !result) {
                LOG_ERROR("Failed to blit texture: {}", result.error());
                framebuffer->unbind();
                return result;
            }
            break;
        }

        default:
            LOG_ERROR("Unknown panel content type: {}", static_cast<int>(panel.content_type));
            framebuffer->unbind();
            return std::unexpected("Unknown panel content type");
        }

        framebuffer->unbind();
        return {};
    }

    Result<RenderResult> SplitViewRenderer::render(
        const SplitViewRequest& request,
        RenderingPipeline& pipeline,
        ScreenQuadRenderer& screen_renderer,
        ManagedShader& quad_shader) {

        LOG_TIMER_TRACE("SplitViewRenderer::render");

        if (!initialized_) {
            if (auto result = initialize(); !result) {
                return std::unexpected("Failed to initialize split view renderer");
            }
        }

        if (request.panels.size() != 2) {
            return std::unexpected("Split view requires exactly 2 panels");
        }

        LOG_DEBUG("Split view: left='{}' (type {}), right='{}' (type {}), split={}",
                  request.panels[0].label, static_cast<int>(request.panels[0].content_type),
                  request.panels[1].label, static_cast<int>(request.panels[1].content_type),
                  request.panels[0].end_position);

        // Always create/resize framebuffers if needed
        if (auto result = createFramebuffers(request.viewport.size.x, request.viewport.size.y); !result) {
            LOG_ERROR("Failed to create framebuffers: {}", result.error());
            return std::unexpected(result.error());
        }

        // Save current OpenGL state
        GLint current_viewport[4];
        glGetIntegerv(GL_VIEWPORT, current_viewport);
        GLint current_fbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);

        // Check if this is a GT comparison (one panel is Image2D or CachedRender)
        bool is_gt_comparison = (request.panels[0].content_type == gs::rendering::PanelContentType::Image2D ||
                                 request.panels[0].content_type == gs::rendering::PanelContentType::CachedRender ||
                                 request.panels[1].content_type == gs::rendering::PanelContentType::Image2D ||
                                 request.panels[1].content_type == gs::rendering::PanelContentType::CachedRender);

        GLuint left_texture = 0;
        GLuint right_texture = 0;

        if (is_gt_comparison) {
            // For GT comparison, use textures directly without rendering to framebuffers
            for (size_t i = 0; i < 2; ++i) {
                const auto& panel = request.panels[i];
                GLuint* target_texture = (i == 0) ? &left_texture : &right_texture;

                if (panel.content_type == gs::rendering::PanelContentType::Image2D ||
                    panel.content_type == gs::rendering::PanelContentType::CachedRender) {
                    // Use the texture directly
                    *target_texture = panel.texture_id;
                    if (*target_texture == 0) {
                        LOG_ERROR("Panel {} has invalid texture ID", i);
                        return std::unexpected("Invalid texture ID");
                    }
                } else if (panel.content_type == gs::rendering::PanelContentType::Model3D) {
                    // Need to render the model - use framebuffer
                    auto* framebuffer = (i == 0) ? left_framebuffer_.get() : right_framebuffer_.get();

                    // Ensure framebuffer exists
                    if (!framebuffer) {
                        LOG_ERROR("Framebuffer for panel {} is null", i);
                        return std::unexpected("Framebuffer not initialized");
                    }

                    if (auto result = renderPanelContent(framebuffer, panel, request,
                                                         pipeline, screen_renderer, quad_shader);
                        !result) {
                        return std::unexpected(result.error());
                    }
                    *target_texture = framebuffer->getFrameTexture();
                }
            }
        } else {
            // For PLY comparison, render both models to framebuffers
            // Ensure framebuffers exist before using them
            if (!left_framebuffer_ || !right_framebuffer_) {
                LOG_ERROR("Framebuffers not initialized for PLY comparison");
                return std::unexpected("Framebuffers not initialized");
            }

            // Render left panel
            if (auto result = renderPanelContent(left_framebuffer_.get(), request.panels[0],
                                                 request, pipeline, screen_renderer, quad_shader);
                !result) {
                return std::unexpected(result.error());
            }
            left_texture = left_framebuffer_->getFrameTexture();

            // Render right panel
            if (auto result = renderPanelContent(right_framebuffer_.get(), request.panels[1],
                                                 request, pipeline, screen_renderer, quad_shader);
                !result) {
                return std::unexpected(result.error());
            }
            right_texture = right_framebuffer_->getFrameTexture();
        }

        // === Composite the two views directly to screen ===
        LOG_DEBUG("Compositing split view at position {}", request.panels[0].end_position);

        // Render directly to screen (framebuffer 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Set viewport to the actual screen area we want to render to
        glViewport(current_viewport[0], current_viewport[1],
                   current_viewport[2], current_viewport[3]);

        // Clear the screen before compositing
        glClearColor(request.background_color.r, request.background_color.g,
                     request.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use split view shader to composite directly to screen
        if (auto result = compositeSplitView(
                left_texture,
                right_texture,
                request.panels[0].end_position,
                request.divider_color,
                request.viewport.size.x);
            !result) {
            LOG_ERROR("Failed to composite split view: {}", result.error());
            glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
            return std::unexpected(result.error());
        }

        // Restore the original framebuffer binding
        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);

        // Return a dummy result
        torch::Tensor dummy_image = torch::zeros({3, request.viewport.size.y, request.viewport.size.x},
                                                 torch::kFloat32)
                                        .to(torch::kCUDA);
        torch::Tensor dummy_depth = torch::empty({0}, torch::kFloat32);

        return RenderResult{
            .image = std::make_shared<torch::Tensor>(std::move(dummy_image)),
            .depth = std::make_shared<torch::Tensor>(std::move(dummy_depth))};
    }

    Result<void> SplitViewRenderer::compositeSplitView(
        GLuint left_texture,
        GLuint right_texture,
        float split_position,
        const glm::vec4& divider_color,
        int viewport_width) {

        LOG_TRACE("Compositing: left_tex={}, right_tex={}, split={}",
                  left_texture, right_texture, split_position);

        // Save current state
        GLint current_viewport[4];
        glGetIntegerv(GL_VIEWPORT, current_viewport);

        // Disable depth test for compositing
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Use the split view shader
        if (auto result = split_shader_.bind(); !result) {
            LOG_ERROR("Failed to bind split shader: {}", result.error());
            return result;
        }

        // Set textures
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, left_texture);
        if (auto result = split_shader_.set("leftTexture", 0); !result) {
            LOG_ERROR("Failed to set leftTexture uniform: {}", result.error());
        }

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, right_texture);
        if (auto result = split_shader_.set("rightTexture", 1); !result) {
            LOG_ERROR("Failed to set rightTexture uniform: {}", result.error());
        }

        // Set other uniforms
        if (auto result = split_shader_.set("splitPosition", split_position); !result) {
            LOG_ERROR("Failed to set splitPosition uniform: {}", result.error());
        }

        // Use float for bool uniform (GLSL converts non-zero to true)
        if (auto result = split_shader_.set("showDivider", 1.0f); !result) {
            LOG_ERROR("Failed to set showDivider uniform: {}", result.error());
        }

        if (auto result = split_shader_.set("dividerColor", divider_color); !result) {
            LOG_ERROR("Failed to set dividerColor uniform: {}", result.error());
        }

        // Calculate normalized divider width (2 pixels for typical viewport)
        float divider_width_pixels = 2.0f;
        float normalized_divider_width = divider_width_pixels / static_cast<float>(viewport_width);

        if (auto result = split_shader_.set("dividerWidth", normalized_divider_width); !result) {
            LOG_ERROR("Failed to set dividerWidth uniform: {}", result.error());
        }

        // Draw the full-screen quad
        VAOBinder vao_bind(quad_vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        LOG_TRACE("Drew composite quad");

        split_shader_.unbind();

        // Restore state
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

        return {};
    }

} // namespace gs::rendering
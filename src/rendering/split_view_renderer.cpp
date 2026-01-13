/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_renderer.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "gl_state_guard.hpp"
#include <glad/glad.h>

namespace lfs::rendering {

    Result<void> SplitViewRenderer::initialize() {
        if (initialized_) {
            return {};
        }

        LOG_DEBUG("Initializing SplitViewRenderer");

        auto shader_result = load_shader("split_view", "split_view.vert", "split_view.frag", false);
        if (!shader_result) {
            LOG_ERROR("Failed to load split view shader: {}", shader_result.error().what());
            return std::unexpected("Failed to load split view shader");
        }
        split_shader_ = std::move(*shader_result);

        auto panel_result = load_shader("split_panel", "screen_quad.vert", "split_panel.frag", false);
        if (!panel_result) {
            LOG_ERROR("Failed to load split panel shader: {}", panel_result.error().what());
            return std::unexpected("Failed to load split panel shader");
        }
        panel_shader_ = std::move(*panel_result);

        auto blit_result = load_shader("texture_blit", "screen_quad.vert", "texture_blit.frag", false);
        if (!blit_result) {
            LOG_WARN("Failed to load texture blit shader, will use quad shader");
        } else {
            texture_blit_shader_ = std::move(*blit_result);
        }

        if (auto result = setupQuad(); !result) {
            return result;
        }

        initialized_ = true;
        LOG_DEBUG("SplitViewRenderer initialized");
        return {};
    }

    Result<void> SplitViewRenderer::setupQuad() {
        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected("Failed to create VAO");
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected("Failed to create VBO");
        }
        quad_vbo_ = std::move(*vbo_result);

        constexpr float QUAD_VERTICES[] = {
            -1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f};

        VAOBuilder builder(std::move(*vao_result));
        std::span<const float> vertices_span(QUAD_VERTICES, 24);

        builder.attachVBO(quad_vbo_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0, .size = 2, .type = GL_FLOAT, .stride = 4 * sizeof(float), .offset = nullptr})
            .setAttribute({.index = 1, .size = 2, .type = GL_FLOAT, .stride = 4 * sizeof(float), .offset = (void*)(2 * sizeof(float))});

        quad_vao_ = builder.build();
        return {};
    }

    Result<void> SplitViewRenderer::createFramebuffers(const int width, const int height) {
        if (!left_framebuffer_) {
            left_framebuffer_ = std::make_unique<FrameBuffer>();
        }
        if (!right_framebuffer_) {
            right_framebuffer_ = std::make_unique<FrameBuffer>();
        }

        if (left_framebuffer_->getWidth() != width || left_framebuffer_->getHeight() != height) {
            left_framebuffer_->resize(width, height);
        }
        if (right_framebuffer_->getWidth() != width || right_framebuffer_->getHeight() != height) {
            right_framebuffer_->resize(width, height);
        }

        return {};
    }

    Result<void> SplitViewRenderer::blitTextureToFramebuffer(const GLuint texture_id) {
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        ManagedShader* shader = texture_blit_shader_.valid() ? &texture_blit_shader_ : nullptr;

        if (shader) {
            if (auto result = shader->bind(); !result) {
                LOG_ERROR("Failed to bind texture blit shader: {}", result.error());
                return result;
            }
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            if (auto result = shader->set("texture0", 0); !result) {
                LOG_TRACE("Failed to set texture0: {}", result.error());
            }

        } else {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_id);
        }

        VAOBinder vao_bind(quad_vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        if (shader) {
            if (auto result = split_shader_.unbind(); !result) {
                LOG_TRACE("Failed to unbind shader: {}", result.error());
            }
        }

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        return {};
    }

    Result<std::optional<RenderingPipeline::RenderResult>> SplitViewRenderer::renderPanelContent(
        FrameBuffer* const framebuffer,
        const SplitViewPanel& panel,
        const SplitViewRequest& request,
        RenderingPipeline& pipeline,
        ScreenQuadRenderer& screen_renderer,
        ManagedShader& quad_shader) {

        framebuffer->bind();

        const glm::ivec2 render_size = request.viewport.size;

        glViewport(0, 0, render_size.x, render_size.y);
        glClearColor(request.background_color.r, request.background_color.g,
                     request.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        switch (panel.content_type) {
        case PanelContentType::Model3D: {
            if (!panel.model) {
                LOG_ERROR("Model3D panel has no model");
                framebuffer->unbind();
                return std::unexpected("Model3D panel has no model");
            }

            RenderingPipeline::RenderRequest base_req{
                .view_rotation = request.viewport.rotation,
                .view_translation = request.viewport.translation,
                .viewport_size = render_size,
                .fov = request.viewport.fov,
                .scaling_modifier = request.scaling_modifier,
                .antialiasing = request.antialiasing,
                .sh_degree = request.sh_degree,
                .render_mode = RenderMode::RGB,
                .crop_box = nullptr,
                .background_color = request.background_color,
                .point_cloud_mode = request.point_cloud_mode,
                .voxel_size = request.voxel_size,
                .gut = request.gut,
                .show_rings = request.show_rings,
                .ring_width = request.ring_width};

            std::unique_ptr<lfs::geometry::BoundingBox> temp_crop_box;
            if (request.crop_box.has_value()) {
                temp_crop_box = std::make_unique<lfs::geometry::BoundingBox>();
                temp_crop_box->setBounds(request.crop_box->min, request.crop_box->max);
                lfs::geometry::EuclideanTransform transform(request.crop_box->transform);
                temp_crop_box->setworld2BBox(transform);
                base_req.crop_box = temp_crop_box.get();
            }

            auto render_result = pipeline.render(*panel.model, base_req);
            if (!render_result) {
                LOG_ERROR("Failed to render model: {}", render_result.error());
                framebuffer->unbind();
                return std::unexpected(render_result.error());
            }

            if (auto upload_result = RenderingPipeline::uploadToScreen(*render_result, screen_renderer, render_size);
                !upload_result) {
                LOG_ERROR("Failed to upload model: {}", upload_result.error());
            } else {
                // Reset GL state for 2D blit
                framebuffer->bind();
                glViewport(0, 0, render_size.x, render_size.y);
                glDisable(GL_DEPTH_TEST);
                glDisable(GL_SCISSOR_TEST);

                if (auto screen_result = screen_renderer.render(panel_shader_); !screen_result) {
                    LOG_ERROR("Failed to render to framebuffer: {}", screen_result.error());
                }

                glEnable(GL_DEPTH_TEST);
            }

            framebuffer->unbind();
            return std::move(*render_result);
        }

        case PanelContentType::Image2D:
        case PanelContentType::CachedRender: {
            if (panel.texture_id == 0) {
                LOG_ERROR("Panel has invalid texture ID");
                framebuffer->unbind();
                return std::unexpected("Panel has invalid texture ID");
            }

            if (auto result = blitTextureToFramebuffer(panel.texture_id); !result) {
                LOG_ERROR("Failed to blit texture: {}", result.error());
                framebuffer->unbind();
                return std::unexpected(result.error());
            }
            break;
        }

        default:
            LOG_ERROR("Unknown panel content type: {}", static_cast<int>(panel.content_type));
            framebuffer->unbind();
            return std::unexpected("Unknown panel content type");
        }

        framebuffer->unbind();
        return std::nullopt;
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

        const int fb_width = request.viewport.size.x;
        const int fb_height = request.viewport.size.y;

        if (auto result = createFramebuffers(fb_width, fb_height); !result) {
            LOG_ERROR("Failed to create framebuffers: {}", result.error());
            return std::unexpected(result.error());
        }

        GLint current_viewport[4];
        glGetIntegerv(GL_VIEWPORT, current_viewport);
        GLint current_fbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);

        const bool is_gt_comparison = (request.panels[0].content_type == PanelContentType::Image2D ||
                                       request.panels[0].content_type == PanelContentType::CachedRender ||
                                       request.panels[1].content_type == PanelContentType::Image2D ||
                                       request.panels[1].content_type == PanelContentType::CachedRender);

        GLuint left_texture = 0;
        GLuint right_texture = 0;
        std::optional<RenderingPipeline::RenderResult> left_render_result;
        std::optional<RenderingPipeline::RenderResult> right_render_result;

        if (is_gt_comparison) {
            for (size_t i = 0; i < 2; ++i) {
                const auto& panel = request.panels[i];
                GLuint* target_texture = (i == 0) ? &left_texture : &right_texture;

                if (panel.content_type == PanelContentType::Image2D ||
                    panel.content_type == PanelContentType::CachedRender) {
                    *target_texture = panel.texture_id;
                    if (*target_texture == 0) {
                        LOG_ERROR("Panel {} has invalid texture ID", i);
                        return std::unexpected("Invalid texture ID");
                    }
                } else if (panel.content_type == PanelContentType::Model3D) {
                    auto* framebuffer = (i == 0) ? left_framebuffer_.get() : right_framebuffer_.get();
                    if (!framebuffer) {
                        LOG_ERROR("Framebuffer for panel {} is null", i);
                        return std::unexpected("Framebuffer not initialized");
                    }

                    auto result = renderPanelContent(framebuffer, panel, request,
                                                     pipeline, screen_renderer, quad_shader);
                    if (!result) {
                        return std::unexpected(result.error());
                    }
                    *target_texture = framebuffer->getFrameTexture();

                    if (result->has_value()) {
                        if (i == 0) {
                            left_render_result = std::move(result->value());
                        } else {
                            right_render_result = std::move(result->value());
                        }
                    }
                }
            }
        } else {
            if (!left_framebuffer_ || !right_framebuffer_) {
                LOG_ERROR("Framebuffers not initialized");
                return std::unexpected("Framebuffers not initialized");
            }

            auto left_panel_result = renderPanelContent(left_framebuffer_.get(), request.panels[0],
                                                        request, pipeline, screen_renderer, quad_shader);
            if (!left_panel_result) {
                return std::unexpected(left_panel_result.error());
            }
            left_texture = left_framebuffer_->getFrameTexture();
            if (left_panel_result->has_value()) {
                left_render_result = std::move(left_panel_result->value());
            }

            auto right_panel_result = renderPanelContent(right_framebuffer_.get(), request.panels[1],
                                                         request, pipeline, screen_renderer, quad_shader);
            if (!right_panel_result) {
                return std::unexpected(right_panel_result.error());
            }
            right_texture = right_framebuffer_->getFrameTexture();
            if (right_panel_result->has_value()) {
                right_render_result = std::move(right_panel_result->value());
            }
        }

        // Composite to screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(current_viewport[0], current_viewport[1],
                   current_viewport[2], current_viewport[3]);

        glClearColor(request.background_color.r, request.background_color.g,
                     request.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const bool flip_left = (request.panels[0].content_type == PanelContentType::Model3D);
        const bool flip_right = (request.panels[1].content_type == PanelContentType::Model3D);

        if (auto result = compositeSplitView(
                left_texture, right_texture,
                request.panels[0].end_position,
                request.left_texcoord_scale, request.right_texcoord_scale,
                request.divider_color, request.viewport.size.x,
                flip_left, flip_right);
            !result) {
            LOG_ERROR("Failed to composite split view: {}", result.error());
            glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
            return std::unexpected(result.error());
        }

        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);

        const auto height = static_cast<size_t>(request.viewport.size.y);
        const auto width = static_cast<size_t>(request.viewport.size.x);

        auto dummy_image = lfs::core::Tensor::zeros({3, height, width}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        RenderResult result{
            .image = std::make_shared<lfs::core::Tensor>(std::move(dummy_image)),
            .depth = left_render_result.has_value()
                         ? std::make_shared<lfs::core::Tensor>(std::move(left_render_result->depth))
                         : std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::Float32)),
            .depth_right = right_render_result.has_value()
                               ? std::make_shared<lfs::core::Tensor>(std::move(right_render_result->depth))
                               : std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::Float32)),
            .valid = true,
            .split_position = request.panels[0].end_position};

        return result;
    }

    Result<void> SplitViewRenderer::compositeSplitView(
        const GLuint left_texture,
        const GLuint right_texture,
        const float split_position,
        const glm::vec2& left_texcoord_scale,
        const glm::vec2& right_texcoord_scale,
        const glm::vec4& divider_color,
        const int viewport_width,
        const bool flip_left_y,
        const bool flip_right_y) {

        constexpr float DIVIDER_WIDTH_PX = 2.0f;

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (auto result = split_shader_.bind(); !result) {
            LOG_ERROR("Failed to bind split shader: {}", result.error());
            return result;
        }

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, left_texture);
        if (auto r = split_shader_.set("leftTexture", 0); !r) {
            return r;
        }

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, right_texture);
        if (auto r = split_shader_.set("rightTexture", 1); !r) {
            return r;
        }

        if (auto result = split_shader_.set("splitPosition", split_position); !result) {
            LOG_TRACE("Uniform 'splitPosition' not found in shader: {}", result.error());
        }

        if (auto result = split_shader_.set("showDivider", true); !result) {
            LOG_TRACE("Uniform 'showDivider' not found in shader: {}", result.error());
        }

        if (auto result = split_shader_.set("dividerColor", divider_color); !result) {
            LOG_TRACE("Uniform 'dividerColor' not found in shader: {}", result.error());
        }

        if (auto result = split_shader_.set("dividerWidth", DIVIDER_WIDTH_PX / static_cast<float>(viewport_width)); !result) {
            LOG_TRACE("Uniform 'dividerWidth' not found in shader: {}", result.error());
        }

        if (auto result = split_shader_.set("leftTexcoordScale", left_texcoord_scale); !result) {
            LOG_TRACE("Uniform 'leftTexcoordScale' not found in shader: {}", result.error());
        }

        if (auto result = split_shader_.set("rightTexcoordScale", right_texcoord_scale); !result) {
            LOG_TRACE("Uniform 'rightTexcoordScale' not found in shader: {}", result.error());
        }

        if (auto result = split_shader_.set("flipLeftY", flip_left_y); !result) {
            LOG_TRACE("Uniform 'flipLeftY' not found in shader: {}", result.error());
        }

        if (auto result = split_shader_.set("flipRightY", flip_right_y); !result) {
            LOG_TRACE("Uniform 'flipRightY' not found in shader: {}", result.error());
        }

        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        split_shader_.set("viewportSize", glm::vec2(static_cast<float>(viewport[2]), static_cast<float>(viewport[3])));

        VAOBinder vao_bind(quad_vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        if (auto result = split_shader_.unbind(); !result) {
            LOG_TRACE("Failed to unbind shader: {}", result.error());
        }
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

        return {};
    }

} // namespace lfs::rendering

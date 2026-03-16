/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_renderer.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "gl_state_guard.hpp"
#include <glad/glad.h>

namespace lfs::rendering {

    namespace {
        [[nodiscard]] RenderResult buildSplitViewMetadata(
            std::optional<RenderingPipeline::RenderResult>& left_render_result,
            std::optional<RenderingPipeline::RenderResult>& right_render_result,
            const float split_position) {
            RenderResult result{
                .depth = (left_render_result && left_render_result->depth.is_valid())
                             ? std::make_shared<lfs::core::Tensor>(std::move(left_render_result->depth))
                             : nullptr,
                .depth_right = (right_render_result && right_render_result->depth.is_valid())
                                   ? std::make_shared<lfs::core::Tensor>(std::move(right_render_result->depth))
                                   : nullptr,
                .valid = true,
                .split_position = split_position};

            if (left_render_result) {
                result.depth_is_ndc = left_render_result->depth_is_ndc;
                result.near_plane = left_render_result->near_plane;
                result.far_plane = left_render_result->far_plane;
                result.orthographic = left_render_result->orthographic;
            } else if (right_render_result) {
                result.depth_is_ndc = right_render_result->depth_is_ndc;
                result.near_plane = right_render_result->near_plane;
                result.far_plane = right_render_result->far_plane;
                result.orthographic = right_render_result->orthographic;
            }

            return result;
        }
    } // namespace

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

    Result<SplitViewRenderer::SplitViewTargets> SplitViewRenderer::acquireTargets(
        RenderTargetPool& render_target_pool,
        const glm::ivec2& size) {

        auto left = render_target_pool.acquire("split_view.left", size);
        if (!left) {
            return std::unexpected(left.error());
        }

        auto right = render_target_pool.acquire("split_view.right", size);
        if (!right) {
            return std::unexpected(right.error());
        }

        auto composite = render_target_pool.acquire("split_view.composite", size);
        if (!composite) {
            return std::unexpected(composite.error());
        }

        return SplitViewTargets{
            .left = std::move(*left),
            .right = std::move(*right),
            .composite = std::move(*composite)};
    }

    Result<void> SplitViewRenderer::blitTextureToFramebuffer(const GLuint texture_id) {
        GLStateGuard state_guard;

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        ManagedShader* shader = texture_blit_shader_.valid() ? &texture_blit_shader_ : nullptr;

        if (shader) {
            ShaderScope scope(*shader);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            if (auto result = shader->set("texture0", 0); !result) {
                LOG_TRACE("Failed to set texture0: {}", result.error());
            }
            VAOBinder vao_bind(quad_vao_);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        } else {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            VAOBinder vao_bind(quad_vao_);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }

        return {};
    }

    Result<std::optional<RenderingPipeline::RenderResult>> SplitViewRenderer::renderPanelContent(
        FrameBuffer& framebuffer,
        const SplitViewPanel& panel,
        const SplitViewRequest& request,
        RenderingPipeline& pipeline,
        ScreenQuadRenderer& screen_renderer,
        ManagedShader& quad_shader) {

        framebuffer.bind();

        const glm::ivec2 render_size = request.viewport.size;

        glViewport(0, 0, render_size.x, render_size.y);
        glClearColor(request.background_color.r, request.background_color.g,
                     request.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        switch (panel.content_type) {
        case PanelContentType::Model3D: {
            if (!panel.model) {
                LOG_ERROR("Model3D panel has no model");
                framebuffer.unbind();
                return std::unexpected("Model3D panel has no model");
            }

            RenderingPipeline::RenderRequest base_req{
                .view_rotation = request.viewport.rotation,
                .view_translation = request.viewport.translation,
                .viewport_size = render_size,
                .focal_length_mm = request.viewport.focal_length_mm,
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
                .ring_width = request.ring_width,
                .model_transforms = {panel.model_transform}};

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
                framebuffer.unbind();
                return std::unexpected(render_result.error());
            }

            if (auto upload_result = RenderingPipeline::uploadToScreen(*render_result, screen_renderer, render_size);
                !upload_result) {
                LOG_ERROR("Failed to upload model: {}", upload_result.error());
            } else {
                // Reset GL state for 2D blit
                framebuffer.bind();
                glViewport(0, 0, render_size.x, render_size.y);
                glDisable(GL_DEPTH_TEST);
                glDisable(GL_SCISSOR_TEST);

                if (auto screen_result = screen_renderer.render(panel_shader_); !screen_result) {
                    LOG_ERROR("Failed to render to framebuffer: {}", screen_result.error());
                }

                glEnable(GL_DEPTH_TEST);
            }

            framebuffer.unbind();
            return std::move(*render_result);
        }

        case PanelContentType::Image2D:
        case PanelContentType::CachedRender: {
            if (panel.texture_id == 0) {
                LOG_ERROR("Panel has invalid texture ID");
                framebuffer.unbind();
                return std::unexpected("Panel has invalid texture ID");
            }

            if (auto result = blitTextureToFramebuffer(panel.texture_id); !result) {
                LOG_ERROR("Failed to blit texture: {}", result.error());
                framebuffer.unbind();
                return std::unexpected(result.error());
            }
            break;
        }

        default:
            LOG_ERROR("Unknown panel content type: {}", static_cast<int>(panel.content_type));
            framebuffer.unbind();
            return std::unexpected("Unknown panel content type");
        }

        framebuffer.unbind();
        return std::nullopt;
    }

    Result<RenderResult> SplitViewRenderer::render(
        const SplitViewRequest& request,
        RenderTargetPool& render_target_pool,
        RenderingPipeline& pipeline,
        ScreenQuadRenderer& screen_renderer,
        ManagedShader& quad_shader) {

        auto frame_result = renderGpuFrame(request, render_target_pool, pipeline, screen_renderer, quad_shader);
        if (!frame_result) {
            return std::unexpected(frame_result.error());
        }

        return std::move(frame_result->metadata);
    }

    Result<SplitViewFrameResult> SplitViewRenderer::renderGpuFrame(
        const SplitViewRequest& request,
        RenderTargetPool& render_target_pool,
        RenderingPipeline& pipeline,
        ScreenQuadRenderer& screen_renderer,
        ManagedShader& quad_shader) {

        LOG_TIMER_TRACE("SplitViewRenderer::renderGpuFrame");

        if (!initialized_) {
            if (auto result = initialize(); !result) {
                return std::unexpected("Failed to initialize split view renderer");
            }
        }

        if (request.panels.size() != 2) {
            return std::unexpected("Split view requires exactly 2 panels");
        }

        auto targets = acquireTargets(render_target_pool, request.viewport.size);
        if (!targets) {
            LOG_ERROR("Failed to acquire split-view targets: {}", targets.error());
            return std::unexpected(targets.error());
        }

        GLFramebufferGuard framebuffer_guard;
        GLViewportGuard viewport_guard;
        GLScissorEnableGuard scissor_guard;

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
                    auto& framebuffer = (i == 0) ? *targets->left : *targets->right;

                    auto result = renderPanelContent(framebuffer, panel, request,
                                                     pipeline, screen_renderer, quad_shader);
                    if (!result) {
                        return std::unexpected(result.error());
                    }
                    *target_texture = framebuffer.getFrameTexture();

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
            auto left_panel_result = renderPanelContent(*targets->left, request.panels[0],
                                                        request, pipeline, screen_renderer, quad_shader);
            if (!left_panel_result) {
                return std::unexpected(left_panel_result.error());
            }
            left_texture = targets->left->getFrameTexture();
            if (left_panel_result->has_value()) {
                left_render_result = std::move(left_panel_result->value());
            }

            auto right_panel_result = renderPanelContent(*targets->right, request.panels[1],
                                                         request, pipeline, screen_renderer, quad_shader);
            if (!right_panel_result) {
                return std::unexpected(right_panel_result.error());
            }
            right_texture = targets->right->getFrameTexture();
            if (right_panel_result->has_value()) {
                right_render_result = std::move(right_panel_result->value());
            }
        }

        targets->composite->bind();
        glViewport(0, 0, request.viewport.size.x, request.viewport.size.y);
        glClearColor(request.background_color.r, request.background_color.g,
                     request.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int composite_x = 0;
        int composite_y = 0;
        int composite_w = request.viewport.size.x;
        int composite_h = request.viewport.size.y;

        if (request.letterbox && request.content_size.x > 0 && request.content_size.y > 0) {
            const float content_aspect = static_cast<float>(request.content_size.x) / request.content_size.y;
            const float viewport_aspect = static_cast<float>(request.viewport.size.x) / request.viewport.size.y;

            if (content_aspect > viewport_aspect) {
                composite_w = request.viewport.size.x;
                composite_h = static_cast<int>(request.viewport.size.x / content_aspect);
                composite_x = 0;
                composite_y = (request.viewport.size.y - composite_h) / 2;
            } else {
                composite_h = request.viewport.size.y;
                composite_w = static_cast<int>(request.viewport.size.y * content_aspect);
                composite_x = (request.viewport.size.x - composite_w) / 2;
                composite_y = 0;
            }
            glViewport(composite_x, composite_y, composite_w, composite_h);
        }

        const bool flip_left = request.flip_left_y.value_or(request.panels[0].content_type == PanelContentType::Model3D);
        const bool flip_right = request.flip_right_y.value_or(request.panels[1].content_type == PanelContentType::Model3D);

        if (auto result = compositeSplitView(
                left_texture, right_texture,
                request.panels[0].end_position,
                request.left_texcoord_scale, request.right_texcoord_scale,
                request.divider_color, composite_w,
                flip_left, flip_right);
            !result) {
            LOG_ERROR("Failed to composite split view: {}", result.error());
            return std::unexpected(result.error());
        }

        SplitViewFrameResult result;
        result.frame = {
            .color = {.id = targets->composite->getFrameTexture(), .size = request.viewport.size},
            .depth = {},
            .depth_is_ndc = false,
            .near_plane = DEFAULT_NEAR_PLANE,
            .far_plane = DEFAULT_FAR_PLANE,
            .orthographic = request.viewport.orthographic};
        result.metadata = buildSplitViewMetadata(
            left_render_result, right_render_result, request.panels[0].end_position);
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

        ShaderScope scope(split_shader_);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, left_texture);
        if (auto r = split_shader_.set("leftTexture", 0); !r)
            return r;

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, right_texture);
        if (auto r = split_shader_.set("rightTexture", 1); !r)
            return r;

        if (auto result = split_shader_.set("splitPosition", split_position); !result)
            LOG_TRACE("Uniform 'splitPosition' not found in shader: {}", result.error());

        if (auto result = split_shader_.set("showDivider", true); !result)
            LOG_TRACE("Uniform 'showDivider' not found in shader: {}", result.error());

        if (auto result = split_shader_.set("dividerColor", divider_color); !result)
            LOG_TRACE("Uniform 'dividerColor' not found in shader: {}", result.error());

        if (auto result = split_shader_.set("dividerWidth", DIVIDER_WIDTH_PX / static_cast<float>(viewport_width)); !result)
            LOG_TRACE("Uniform 'dividerWidth' not found in shader: {}", result.error());

        if (auto result = split_shader_.set("leftTexcoordScale", left_texcoord_scale); !result)
            LOG_TRACE("Uniform 'leftTexcoordScale' not found in shader: {}", result.error());

        if (auto result = split_shader_.set("rightTexcoordScale", right_texcoord_scale); !result)
            LOG_TRACE("Uniform 'rightTexcoordScale' not found in shader: {}", result.error());

        if (auto result = split_shader_.set("flipLeftY", flip_left_y); !result)
            LOG_TRACE("Uniform 'flipLeftY' not found in shader: {}", result.error());

        if (auto result = split_shader_.set("flipRightY", flip_right_y); !result)
            LOG_TRACE("Uniform 'flipRightY' not found in shader: {}", result.error());

        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        split_shader_.set("viewportSize", glm::vec2(static_cast<float>(viewport[2]), static_cast<float>(viewport[3])));

        VAOBinder vao_bind(quad_vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

        return {};
    }

} // namespace lfs::rendering

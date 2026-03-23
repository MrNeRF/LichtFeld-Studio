/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_renderer.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "gl_state_guard.hpp"
#include <format>
#include <glad/glad.h>
#include <vector>

namespace lfs::rendering {

    namespace {
        [[nodiscard]] FrameMetadata buildSplitViewMetadata(
            std::optional<FrameMetadata>& left_render_metadata,
            std::optional<FrameMetadata>& right_render_metadata,
            const float split_position) {
            FrameMetadata result{
                .depth = left_render_metadata ? std::move(left_render_metadata->depth) : nullptr,
                .depth_right = right_render_metadata ? std::move(right_render_metadata->depth) : nullptr,
                .valid = true,
                .split_position = split_position};

            if (left_render_metadata) {
                result.depth_is_ndc = left_render_metadata->depth_is_ndc;
                result.external_depth_texture = left_render_metadata->external_depth_texture;
                result.near_plane = left_render_metadata->near_plane;
                result.far_plane = left_render_metadata->far_plane;
                result.orthographic = left_render_metadata->orthographic;
            } else if (right_render_metadata) {
                result.depth_is_ndc = right_render_metadata->depth_is_ndc;
                result.external_depth_texture = right_render_metadata->external_depth_texture;
                result.near_plane = right_render_metadata->near_plane;
                result.far_plane = right_render_metadata->far_plane;
                result.orthographic = right_render_metadata->orthographic;
            }

            return result;
        }

        [[nodiscard]] RenderingPipeline::ImageRenderResult makeUploadResult(
            const std::shared_ptr<Tensor>& image,
            const FrameMetadata& metadata) {
            return RenderingPipeline::ImageRenderResult{
                .image = image ? *image : Tensor(),
                .depth = metadata.depth ? *metadata.depth : Tensor(),
                .valid = image && image->is_valid(),
                .depth_is_ndc = metadata.depth_is_ndc,
                .external_depth_texture = metadata.external_depth_texture,
                .near_plane = metadata.near_plane,
                .far_plane = metadata.far_plane,
                .orthographic = metadata.orthographic};
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
        auto composite = render_target_pool.acquire("split_view.composite", size);
        if (!composite) {
            return std::unexpected(composite.error());
        }

        return SplitViewTargets{
            .composite = std::move(*composite)};
    }

    Result<ScreenQuadRenderer*> SplitViewRenderer::ensurePanelUploadRenderer(const size_t panel_index) {
        if (panel_index >= panel_upload_renderers_.size()) {
            return std::unexpected("Split-view panel index out of range");
        }

        auto& renderer = panel_upload_renderers_[panel_index];
        if (!renderer) {
            try {
                renderer = std::make_unique<ScreenQuadRenderer>(getPreferredFrameBufferMode());
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Failed to create split-view panel renderer: {}", e.what()));
            }
        }

        return renderer.get();
    }

    Result<SplitViewRenderer::PanelRenderOutput> SplitViewRenderer::renderPanelContent(
        const size_t panel_index,
        const SplitViewPanel& panel,
        const glm::ivec2& output_size,
        RenderingEngine& engine) {

        switch (panel.content.type) {
        case PanelContentType::Model3D: {
            if (!panel.content.model) {
                LOG_ERROR("Model3D panel has no model");
                return std::unexpected("Model3D panel has no model");
            }

            const std::vector<glm::mat4> model_transforms{panel.content.model_transform};

            RenderingPipeline::ImageRenderResult upload_result;
            FrameMetadata metadata;
            if (panel.content.point_cloud_render.has_value()) {
                const auto& render_state = *panel.content.point_cloud_render;
                PointCloudRenderRequest point_cloud_request{
                    .frame_view = render_state.frame_view,
                    .render = render_state.render,
                    .scene = {.model_transforms = &model_transforms},
                    .filters = render_state.filters};

                auto render_result = engine.renderPointCloudImage(*panel.content.model, point_cloud_request);
                if (!render_result) {
                    LOG_ERROR("Failed to render point-cloud split-view panel {}: {}", panel_index, render_result.error());
                    return std::unexpected(render_result.error());
                }

                metadata = render_result->metadata;
                upload_result = makeUploadResult(render_result->image, render_result->metadata);
            } else {
                if (!panel.content.gaussian_render.has_value()) {
                    LOG_ERROR("Model3D split-view panel {} has no render state", panel_index);
                    return std::unexpected("Model3D split-view panel has no render state");
                }

                const auto& render_state = *panel.content.gaussian_render;
                ViewportRenderRequest gaussian_request{
                    .frame_view = render_state.frame_view,
                    .scaling_modifier = render_state.scaling_modifier,
                    .antialiasing = render_state.antialiasing,
                    .mip_filter = render_state.mip_filter,
                    .sh_degree = render_state.sh_degree,
                    .gut = render_state.gut,
                    .equirectangular = render_state.equirectangular,
                    .scene =
                        {.model_transforms = &model_transforms,
                         .transform_indices = nullptr,
                         .node_visibility_mask = {}},
                    .filters = render_state.filters,
                    .overlay = render_state.overlay};

                auto render_result = engine.renderGaussiansImage(*panel.content.model, gaussian_request);
                if (!render_result) {
                    LOG_ERROR("Failed to render gaussian split-view panel {}: {}", panel_index, render_result.error());
                    return std::unexpected(render_result.error());
                }

                metadata = render_result->metadata;
                upload_result = makeUploadResult(render_result->image, render_result->metadata);
            }

            auto upload_renderer = ensurePanelUploadRenderer(panel_index);
            if (!upload_renderer) {
                return std::unexpected(upload_renderer.error());
            }

            if (auto result = RenderingPipeline::uploadToScreen(upload_result, **upload_renderer, output_size);
                !result) {
                LOG_ERROR("Failed to upload model panel {}: {}", panel_index, result.error());
                return std::unexpected(result.error());
            }

            return PanelRenderOutput{
                .texture_id = (*upload_renderer)->getUploadedColorTexture(),
                .texcoord_scale = (*upload_renderer)->getTexcoordScale(),
                .metadata = std::move(metadata),
                .flip_y = false};
        }

        case PanelContentType::Image2D:
        case PanelContentType::CachedRender: {
            if (panel.content.texture_id == 0) {
                LOG_ERROR("Panel has invalid texture ID");
                return std::unexpected("Panel has invalid texture ID");
            }
            return PanelRenderOutput{
                .texture_id = panel.content.texture_id,
                .texcoord_scale = panel.presentation.texcoord_scale,
                .metadata = std::nullopt,
                .flip_y = false};
        }

        default:
            LOG_ERROR("Unknown panel content type: {}", static_cast<int>(panel.content.type));
            return std::unexpected("Unknown panel content type");
        }
    }

    Result<SplitViewFrameResult> SplitViewRenderer::renderGpuFrame(
        const SplitViewRequest& request,
        RenderTargetPool& render_target_pool,
        RenderingEngine& engine) {

        LOG_TIMER_TRACE("SplitViewRenderer::renderGpuFrame");

        if (!initialized_) {
            if (auto result = initialize(); !result) {
                return std::unexpected("Failed to initialize split view renderer");
            }
        }

        auto targets = acquireTargets(render_target_pool, request.composite.output_size);
        if (!targets) {
            LOG_ERROR("Failed to acquire split-view targets: {}", targets.error());
            return std::unexpected(targets.error());
        }

        GLFramebufferGuard framebuffer_guard;
        GLViewportGuard viewport_guard;
        GLScissorEnableGuard scissor_guard;
        // Split-view rendering composites into offscreen FBOs. If the GUI viewport scissor
        // remains enabled here, the window-space scissor box can clip the entire offscreen draw.
        glDisable(GL_SCISSOR_TEST);

        std::array<PanelRenderOutput, 2> panel_outputs;
        for (size_t i = 0; i < request.panels.size(); ++i) {
            auto panel_result = renderPanelContent(i, request.panels[i], request.composite.output_size, engine);
            if (!panel_result) {
                return std::unexpected(panel_result.error());
            }
            panel_outputs[i] = std::move(*panel_result);
        }

        std::optional<FrameMetadata> left_render_metadata =
            std::move(panel_outputs[0].metadata);
        std::optional<FrameMetadata> right_render_metadata =
            std::move(panel_outputs[1].metadata);
        FrameMetadata metadata = buildSplitViewMetadata(
            left_render_metadata, right_render_metadata, request.panels[0].presentation.end_position);

        targets->composite->bind();
        glViewport(0, 0, request.composite.output_size.x, request.composite.output_size.y);
        glClearColor(request.composite.background_color.r, request.composite.background_color.g,
                     request.composite.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int composite_x = 0;
        int composite_y = 0;
        int composite_w = request.composite.output_size.x;
        int composite_h = request.composite.output_size.y;

        if (request.presentation.letterbox &&
            request.presentation.content_size.x > 0 &&
            request.presentation.content_size.y > 0) {
            const float content_aspect = static_cast<float>(request.presentation.content_size.x) /
                                         request.presentation.content_size.y;
            const float viewport_aspect = static_cast<float>(request.composite.output_size.x) /
                                          request.composite.output_size.y;

            if (content_aspect > viewport_aspect) {
                composite_w = request.composite.output_size.x;
                composite_h = static_cast<int>(request.composite.output_size.x / content_aspect);
                composite_x = 0;
                composite_y = (request.composite.output_size.y - composite_h) / 2;
            } else {
                composite_h = request.composite.output_size.y;
                composite_w = static_cast<int>(request.composite.output_size.y * content_aspect);
                composite_x = (request.composite.output_size.x - composite_w) / 2;
                composite_y = 0;
            }
            glViewport(composite_x, composite_y, composite_w, composite_h);
        }

        const bool flip_left = request.panels[0].presentation.flip_y.value_or(panel_outputs[0].flip_y);
        const bool flip_right = request.panels[1].presentation.flip_y.value_or(panel_outputs[1].flip_y);

        if (auto result = compositeSplitView(
                panel_outputs[0].texture_id, panel_outputs[1].texture_id,
                request.panels[0].presentation.end_position,
                panel_outputs[0].texcoord_scale,
                panel_outputs[1].texcoord_scale,
                flip_left, flip_right);
            !result) {
            LOG_ERROR("Failed to composite split view: {}", result.error());
            return std::unexpected(result.error());
        }

        SplitViewFrameResult result;
        result.frame = {
            .color = {.id = targets->composite->getFrameTexture(), .size = request.composite.output_size},
            .depth = {},
            .depth_is_ndc = metadata.depth_is_ndc,
            .near_plane = metadata.near_plane,
            .far_plane = metadata.far_plane,
            .orthographic = metadata.orthographic};
        result.metadata = std::move(metadata);
        return result;
    }

    Result<void> SplitViewRenderer::compositeSplitView(
        const GLuint left_texture,
        const GLuint right_texture,
        const float split_position,
        const glm::vec2& left_texcoord_scale,
        const glm::vec2& right_texcoord_scale,
        const bool flip_left_y,
        const bool flip_right_y) {

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

        if (auto result = split_shader_.set("showDivider", false); !result)
            LOG_TRACE("Uniform 'showDivider' not found in shader: {}", result.error());

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
